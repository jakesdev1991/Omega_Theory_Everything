# syntax=docker/dockerfile:1
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.27.0

ENV ELAN_HOME=/opt/elan
ENV PATH=/opt/elan/bin:${PATH}
ENV LEAN_VERIFIER_ROOT=/opt/SafeVerify

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl git python3 zstd xz-utils build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install a pinned Lean toolchain. Mathlib and SafeVerify are also pinned by the
# vendored lake manifest, so runtime verification does not use a moving version.
RUN curl --proto '=https' --tlsv1.2 -sSf \
      https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
      | sh -s -- -y --no-modify-path --default-toolchain "${LEAN_TOOLCHAIN}" \
    && elan default "${LEAN_TOOLCHAIN}" \
    && lean --version \
    && lake --version

COPY vendor/SafeVerify /opt/SafeVerify
WORKDIR /opt/SafeVerify

# Fetch the exact revisions in lake-manifest.json, download Mathlib's binary
# cache, and build the kernel-replay verifier.
RUN lake update \
    && lake exe cache get \
    && lake build safe_verify \
    && test -x .lake/build/bin/safe_verify

COPY docker/worker.py /opt/lean-verifier-worker.py
RUN chmod 0555 /opt/lean-verifier-worker.py \
    && chmod -R a+rX /opt/elan /opt/SafeVerify \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin verifier

USER 10001:10001
ENTRYPOINT ["python3", "/opt/lean-verifier-worker.py"]
