package org.example.processor;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.TestInputTopic;
import org.apache.kafka.streams.TestOutputTopic;
import org.apache.kafka.streams.Topology;
import org.apache.kafka.streams.TopologyTestDriver;
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.StoreBuilder;
import org.apache.kafka.streams.state.Stores;
import org.junit.jupiter.api.Test;

import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeatureStateProcessorTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String INPUT_TOPIC = "feature-patches";
    private static final String OUTPUT_TOPIC = "feature-state-updates";
    private static final String STORE_NAME = "feature-state-store";

    @Test
    void emitsCanonicalPartialSnapshotFromSinglePatch() throws Exception {
        try (TopologyTestDriver driver = new TopologyTestDriver(topology(), props())) {
            TestInputTopic<String, String> input = inputTopic(driver);
            TestOutputTopic<String, String> output = outputTopic(driver);

            input.pipeInput("secom-0000001", featurePatch("secom-0000001", 10.0, """
                "f000": 1.2,
                "f001": null
                """));

            KeyValue<String, String> result = output.readKeyValue();
            JsonNode snapshot = MAPPER.readTree(result.value);
            JsonNode features = snapshot.get("features");

            assertEquals("secom-0000001", result.key);
            assertEquals("secom-0000001", snapshot.get("sample_id").asText());
            assertEquals("partial", snapshot.get("snapshot_status").asText());
            assertEquals(2, snapshot.get("feature_count").asInt());
            assertEquals(589, snapshot.get("missing_count").asInt());
            assertFalse(snapshot.get("is_complete").asBoolean());
            assertEquals(1, snapshot.get("source_event_count").asInt());
            assertEquals(1, snapshot.get("snapshot_version").asInt());
            assertEquals(
                snapshot.get("source_event_count").asInt(),
                snapshot.get("snapshot_version").asInt()
            );
            assertEquals(10.0, snapshot.get("window_start").asDouble());
            assertEquals(10.0, snapshot.get("window_end").asDouble());
            assertEquals("early", snapshot.get("last_feature_group").asText());
            assertEquals(590, features.size());
            assertEquals(1.2, features.get("f000").asDouble());
            assertTrue(features.get("f001").isNull());
            assertTrue(features.get("f589").isNull());
        }
    }

    @Test
    void mergesMultiplePatchesIntoExistingState() throws Exception {
        try (TopologyTestDriver driver = new TopologyTestDriver(topology(), props())) {
            TestInputTopic<String, String> input = inputTopic(driver);
            TestOutputTopic<String, String> output = outputTopic(driver);

            input.pipeInput("secom-0000001", featurePatch("secom-0000001", 10.0, "\"f000\": 1.2"));
            JsonNode firstSnapshot = MAPPER.readTree(output.readValue());

            assertEquals(10.0, firstSnapshot.get("snapshot_time").asDouble());
            assertEquals(1, firstSnapshot.get("source_event_count").asInt());
            assertEquals(1, firstSnapshot.get("snapshot_version").asInt());
            assertEquals(
                firstSnapshot.get("source_event_count").asInt(),
                firstSnapshot.get("snapshot_version").asInt()
            );

            input.pipeInput("secom-0000001", featurePatch("secom-0000001", 8.0, "\"f001\": 2.3"));

            KeyValue<String, String> result = output.readKeyValue();
            JsonNode snapshot = MAPPER.readTree(result.value);
            JsonNode features = snapshot.get("features");

            assertEquals("secom-0000001", result.key);
            assertEquals(2, snapshot.get("feature_count").asInt());
            assertEquals(588, snapshot.get("missing_count").asInt());
            assertEquals(2, snapshot.get("source_event_count").asInt());
            assertEquals(2, snapshot.get("snapshot_version").asInt());
            assertEquals(
                snapshot.get("source_event_count").asInt(),
                snapshot.get("snapshot_version").asInt()
            );
            assertTrue(snapshot.get("snapshot_version").asInt() > firstSnapshot.get("snapshot_version").asInt());
            assertEquals(10.0, snapshot.get("snapshot_time").asDouble());
            assertEquals(8.0, snapshot.get("window_start").asDouble());
            assertEquals(10.0, snapshot.get("window_end").asDouble());
            assertEquals("state:secom-0000001:10000:2", snapshot.get("serving_snapshot_id").asText());
            assertEquals(1.2, features.get("f000").asDouble());
            assertEquals(2.3, features.get("f001").asDouble());
        }
    }

    private static Topology topology() {
        StoreBuilder<KeyValueStore<String, String>> storeBuilder = Stores.keyValueStoreBuilder(
            Stores.inMemoryKeyValueStore(STORE_NAME),
            Serdes.String(),
            Serdes.String()
        );

        Topology topology = new Topology();
        topology.addSource(
            "feature-patch-source",
            Serdes.String().deserializer(),
            Serdes.String().deserializer(),
            INPUT_TOPIC
        );
        topology.addProcessor(
            "feature-state-processor",
            () -> new FeatureStateProcessor(STORE_NAME),
            "feature-patch-source"
        );
        topology.addStateStore(storeBuilder, "feature-state-processor");
        topology.addSink(
            "feature-state-sink",
            OUTPUT_TOPIC,
            Serdes.String().serializer(),
            Serdes.String().serializer(),
            "feature-state-processor"
        );
        return topology;
    }

    private static Properties props() {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "feature-state-processor-test");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "dummy:9092");
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.StringSerde.class.getName());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.StringSerde.class.getName());
        return props;
    }

    private static TestInputTopic<String, String> inputTopic(TopologyTestDriver driver) {
        return driver.createInputTopic(
            INPUT_TOPIC,
            Serdes.String().serializer(),
            Serdes.String().serializer()
        );
    }

    private static TestOutputTopic<String, String> outputTopic(TopologyTestDriver driver) {
        return driver.createOutputTopic(
            OUTPUT_TOPIC,
            Serdes.String().deserializer(),
            Serdes.String().deserializer()
        );
    }

    private static String featurePatch(String sampleId, double eventTime, String featuresJson) {
        return """
            {
              "event_id": "evt-001",
              "sample_id": "%s",
              "event_time": %.1f,
              "feature_group": "early",
              "features": {
                %s
              },
              "simulation_run_id": "sim-001",
              "drift_segment": "stable"
            }
            """.formatted(sampleId, eventTime, featuresJson);
    }
}
