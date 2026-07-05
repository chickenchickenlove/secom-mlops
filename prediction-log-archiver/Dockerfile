FROM gradle:8.14.3-jdk17 AS build

WORKDIR /app
COPY settings.gradle build.gradle ./
COPY src ./src

RUN gradle installDist --no-daemon

FROM eclipse-temurin:17-jre

WORKDIR /app
COPY --from=build /app/build/install/prediction-log-archiver /app

ENTRYPOINT ["/app/bin/prediction-log-archiver"]
