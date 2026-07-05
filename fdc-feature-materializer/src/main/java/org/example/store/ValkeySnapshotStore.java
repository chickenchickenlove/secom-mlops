package org.example.store;

import org.example.config.MaterializerConfig;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

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
    public void put(String key, String value) {
        try (Jedis jedis = jedisPool.getResource()) {
            String result = jedis.set(key, value);

            if (!"OK".equalsIgnoreCase(result)) {
                throw new IllegalStateException("Valkey SET failed: key=" + key + " result=" + result);
            }

            if (verifyWrite) {
                String storedValue = jedis.get(key);
                if (!value.equals(storedValue)) {
                    throw new IllegalStateException("Valkey read-after-write verification failed: key=" + key);
                }
            }
        }
    }

    @Override
    public void close() {
        jedisPool.close();
    }

    private static JedisPool buildJedisPool(MaterializerConfig config) {
        JedisPoolConfig poolConfig = new JedisPoolConfig();
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
}
