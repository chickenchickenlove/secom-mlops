package org.example;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.consumer.OffsetAndMetadata;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.errors.WakeupException;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

public final class LabelArchiverApp {
    private static final Logger LOG = LoggerFactory.getLogger(LabelArchiverApp.class);

    private final LabelArchiverConfig config;
    private final LabelRepository repository;
    private final AtomicBoolean running = new AtomicBoolean(true);

    private LabelArchiverApp(LabelArchiverConfig config) {
        this.config = config;
        this.repository = new LabelRepository();
    }

    public static void main(String[] args) {
        LabelArchiverConfig config = LabelArchiverConfig.fromEnv();
        new LabelArchiverApp(config).run();
    }

    private void run() {
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(consumerProperties());

        try (consumer; Connection connection = openConnection()) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                running.set(false);
                consumer.wakeup();
            }));

            connection.setAutoCommit(false);
            verifyDatabaseConnection(connection);

            consumer.subscribe(List.of(config.topic()));

            LOG.info(
                "label_archiver_started topic={} group_id={} bootstrap_servers={} db_url={}",
                config.topic(),
                config.groupId(),
                config.bootstrapServers(),
                config.dbUrl()
            );

            while (running.get()) {
                ConsumerRecords<String, String> records =
                    consumer.poll(Duration.ofMillis(config.pollTimeoutMs()));

                for (ConsumerRecord<String, String> record : records) {
                    archiveThenCommit(consumer, connection, record);
                }
            }
        } catch (WakeupException error) {
            if (running.get()) {
                throw error;
            }
        } catch (SQLException error) {
            throw new RuntimeException("label archiver database failure", error);
        }
    }

    private Properties consumerProperties() {
        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, config.bootstrapServers());
        props.put(ConsumerConfig.GROUP_ID_CONFIG, config.groupId());
        props.put(ConsumerConfig.CLIENT_ID_CONFIG, config.clientId());
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, config.autoOffsetReset());
        props.put(ConsumerConfig.ISOLATION_LEVEL_CONFIG, "read_committed");
        props.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, Integer.toString(config.maxPollRecords()));
        props.put("broker.address.family", "v4");
        return props;
    }

    private Connection openConnection() throws SQLException {
        return DriverManager.getConnection(config.dbUrl(), config.dbUser(), config.dbPassword());
    }

    private static void verifyDatabaseConnection(Connection connection) throws SQLException {
        try (PreparedStatement statement = connection.prepareStatement("SELECT 1")) {
            statement.executeQuery();
        }
    }

    private void archiveThenCommit(
        KafkaConsumer<String, String> consumer,
        Connection connection,
        ConsumerRecord<String, String> record
    ) throws SQLException {
        LabelRow label = LabelParser.parse(record.value(), record.key());

        try {
            repository.upsert(connection, label);
            connection.commit();
        } catch (SQLException error) {
            rollbackQuietly(connection);
            throw error;
        }

        TopicPartition topicPartition = new TopicPartition(record.topic(), record.partition());

        consumer.commitSync(Map.of(
            topicPartition,
            new OffsetAndMetadata(record.offset() + 1)
        ));

        LOG.info(
            "label_archived sample_id={} actual_label={} topic={} partition={} offset={} committed_offset={}",
            label.sampleId(),
            label.actualLabel(),
            record.topic(),
            record.partition(),
            record.offset(),
            record.offset() + 1
        );
    }

    private static void rollbackQuietly(Connection connection) {
        try {
            connection.rollback();
        } catch (SQLException rollbackError) {
            LOG.error("label_archiver_rollback_failed", rollbackError);
        }
    }
}
