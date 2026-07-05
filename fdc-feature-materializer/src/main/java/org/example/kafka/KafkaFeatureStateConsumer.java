package org.example.kafka;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.consumer.OffsetAndMetadata;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.example.config.MaterializerConfig;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Properties;

public final class KafkaFeatureStateConsumer implements AutoCloseable {
    private final KafkaConsumer<String, String> consumer;
    private final String topic;
    private final Duration pollTimeout;

    public KafkaFeatureStateConsumer(MaterializerConfig config) {
        this.consumer = new KafkaConsumer<>(consumerProperties(config));
        this.topic = config.topic();
        this.pollTimeout = Duration.ofMillis(config.pollTimeoutMs());
    }

    public void subscribe() {
        consumer.subscribe(List.of(topic));
    }

    public ConsumerRecords<String, String> poll() {
        return consumer.poll(pollTimeout);
    }

    public void commit(ConsumerRecord<String, String> record) {
        TopicPartition topicPartition = new TopicPartition(record.topic(), record.partition());

        consumer.commitSync(Map.of(
            topicPartition,
            new OffsetAndMetadata(record.offset() + 1)
        ));
    }

    public void wakeup() {
        consumer.wakeup();
    }

    @Override
    public void close() {
        consumer.close();
    }

    private static Properties consumerProperties(MaterializerConfig config) {
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
}
