FROM curlimages/curl:8.11.1 AS jmx-exporter

ARG JMX_EXPORTER_VERSION=1.0.1

RUN curl -fsSL \
    -o /tmp/jmx_prometheus_javaagent.jar \
    "https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/${JMX_EXPORTER_VERSION}/jmx_prometheus_javaagent-${JMX_EXPORTER_VERSION}.jar"

FROM apache/kafka:4.3.1

COPY --from=jmx-exporter /tmp/jmx_prometheus_javaagent.jar /opt/jmx-exporter/jmx_prometheus_javaagent.jar
COPY kafka/jmx-exporter.yml /opt/jmx-exporter/kafka-jmx-exporter.yml
