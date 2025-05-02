CREATE SERVICE agent_gateway_that_works
  IN COMPUTE POOL  TUTORIAL_COMPUTE_POOL
  FROM SPECIFICATION $$
spec:
  containers:
    - name: agent-gateway
      image: /cube_testing/public/awesome_images/of
      env:
        SNOWFLAKE_WAREHOUSE: WH_XS
  endpoints:
    - name: queryendpoint
      port: 80
      public: true
  $$;
