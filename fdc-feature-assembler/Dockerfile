FROM eclipse-temurin:17-jdk AS build

WORKDIR /workspace
COPY gradle ./gradle
COPY gradlew build.gradle settings.gradle ./
RUN chmod +x ./gradlew
COPY src ./src
RUN ./gradlew --no-daemon installDist

FROM eclipse-temurin:17-jre

WORKDIR /app
COPY --from=build /workspace/build/install/fdc-feature-assembler/ ./
ENTRYPOINT ["./bin/fdc-feature-assembler"]

