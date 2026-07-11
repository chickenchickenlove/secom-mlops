package org.example.store;

import org.example.config.MaterializerConfig;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

import java.time.Instant;
import java.util.function.BiFunction;
import java.util.function.DoubleSupplier;
import java.util.function.Function;

public final class ValkeySnapshotStore implements SnapshotStore {
    private final JedisPool jedisPool;
    private final boolean verifyWrite;

    public ValkeySnapshotStore(MaterializerConfig config) {
        this.jedisPool = buildJedisPool(config);
        this.verifyWrite = config.verifyWrite();
    }

    @Override
    public void verifyConnection() {
        try (Jedis jedis = jedisPool.getResource()) {
            String pong = jedis.ping();
            if (!"PONG".equalsIgnoreCase(pong)) {
                throw new IllegalStateException("Valkey ping failed: " + pong);
            }
        }
    }

    @Override
    public SnapshotWriteResult put(String key, String value) {
        try (Jedis jedis = jedisPool.getResource()) {
            return writeAndVerify(
                key,
                value,
                verifyWrite,
                jedis::set,
                jedis::get,
                ValkeySnapshotStore::currentEpochSeconds
            );
        }
    }

    @Override
    public void close() {
        jedisPool.close();
    }

    private static JedisPool buildJedisPool(MaterializerConfig config) {
        final JedisPoolConfig poolConfig = new JedisPoolConfig();
        poolConfig.setMaxTotal(config.valkeyPoolMaxTotal());
        poolConfig.setMaxIdle(config.valkeyPoolMaxIdle());
        poolConfig.setMinIdle(0);

        return new JedisPool(
            poolConfig,
            config.valkeyHost(),
            config.valkeyPort(),
            config.valkeyTimeoutMs(),
            null,
            config.valkeyDatabase()
        );
    }

    static SnapshotWriteResult writeAndVerify(
        String key,
        String value,
        boolean verifyWrite,
        BiFunction<String, String, String> setter,
        Function<String, String> getter,
        DoubleSupplier epochSeconds
    ) {
        final String result = setter.apply(key, value);

        if (!"OK".equalsIgnoreCase(result)) {
            throw new IllegalStateException("Valkey SET failed: key=" + key + " result=" + result);
        }

        final SnapshotWriteResult writeResult = new SnapshotWriteResult(epochSeconds.getAsDouble());

        if (verifyWrite) {
            String storedValue = getter.apply(key);
            if (!value.equals(storedValue)) {
                throw new IllegalStateException("Valkey read-after-write verification failed: key=" + key);
            }
        }

        return writeResult;
    }

    private static double currentEpochSeconds() {
        final Instant now = Instant.now();
        return now.getEpochSecond() + now.getNano() / 1_000_000_000.0;
    }
}
