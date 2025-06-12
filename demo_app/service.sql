CREATE SERVICE agent_gateway_that_works
  IN COMPUTE POOL tutorial_compute_pool
  FROM SPECIFICATION $$
    spec:
      containers:
      - name: agent-gateway
        image: /cube_testing/public/awesome_images/of:latest
        env:
          SERVER_PORT: 8080
          SNOWFLAKE_WAREHOUSE: WH_XS
        readinessProbe:
          port: 8080
          path: /healthcheck
      endpoints:
      - name: echoendpoint
        port: 8080
        public: true
      $$
   MIN_INSTANCES=1
   MAX_INSTANCES=1;
