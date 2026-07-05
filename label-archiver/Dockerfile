FROM eclipse-temurin:17-jdk AS build

WORKDIR /workspace
COPY gradle ./gradle
COPY gradlew build.gradle settings.gradle ./
RUN chmod +x ./gradlew
COPY src ./src
RUN ./gradlew --no-daemon installDist

FROM eclipse-temurin:17-jre

WORKDIR /app
COPY --from=build /workspace/build/install/label-archiver/ ./
ENTRYPOINT ["./bin/label-archiver"]