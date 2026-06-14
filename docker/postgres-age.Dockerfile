# OPT-IN: one Postgres that does ALL THREE — vectors (pgvector), graph (Apache
# AGE), and relational SQL. The default compose uses the stock pgvector image
# (vectors + relational), which is everything v1 needs. Switch to this image
# when the graph layer lands (entity relationships, design-doc "Month 2").
#
# Build:  docker build -f docker/postgres-age.Dockerfile -t memgram/postgres-age .
FROM pgvector/pgvector:pg16

ENV AGE_VERSION=PG16/v1.5.0-rc0

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        postgresql-server-dev-16 flex bison; \
    git clone --depth 1 --branch "${AGE_VERSION}" https://github.com/apache/age.git /tmp/age; \
    cd /tmp/age; make && make install; \
    rm -rf /tmp/age; \
    apt-get purge -y --auto-remove build-essential git flex bison; \
    rm -rf /var/lib/apt/lists/*

# AGE must be preloaded. pgvector loads on CREATE EXTENSION; AGE needs this.
RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample
