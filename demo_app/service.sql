CREATE SERVICE agent_gateway
  IN COMPUTE POOL tutorial_compute_pool
  FROM SPECIFICATION $$
    spec:
      containers:
      - name: agent-gateway
        image: /cube_testing/public/image_repo/of:latest
        env:
          SERVER_PORT: 8000
          SNOWFLAKE_WAREHOUSE: WH_XS
        readinessProbe:
          port: 8000
          path: /healthcheck
      endpoints:
      - name: echoendpoint
        port: 8000
        public: true
      $$
   MIN_INSTANCES=1
   MAX_INSTANCES=1;
