API Documentation
================

Task Management
--------------

.. http:post:: /api/tasks

   Create a new task.

   **Example request**:

   .. sourcecode:: http

      POST /api/tasks HTTP/1.1
      Content-Type: application/json

      {
          "type": "content_generation",
          "parameters": {
              "keyword": "example keyword",
              "intent": "informational"
          },
          "priority": "high"
      }

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
          "success": true,
          "task_id": "123456",
          "status": "queued"
      }

Agent Communication
------------------

.. http:get:: /api/communication/flows

   Get inter-agent communication flows.

   :query start_time: Start time for filtering (ISO format)
   :query end_time: End time for filtering (ISO format)
   :statuscode 200: No error

.. http:get:: /api/communication/realtime

   Stream real-time agent interaction data.

   :statuscode 200: Server-sent events stream started

.. http:get:: /api/communication/analytics

   Get communication patterns analytics.

   :query timeframe: Time frame for analysis (e.g., '24h')
   :statuscode 200: No error

Process Timeline
---------------

.. http:get:: /api/timeline/process

   Get detailed process stage tracking.

   :query process_id: ID of the process to track
   :query include_details: Include additional details (boolean)
   :statuscode 200: No error

.. http:get:: /api/timeline/decisions

   Get decision point logging.

   :query process_id: ID of the process
   :query start_time: Start time for filtering
   :query end_time: End time for filtering
   :statuscode 200: No error

Resource Monitoring
------------------

.. http:get:: /api/resources/allocation

   Get detailed resource allocation tracking.

   :query resource_type: Type of resource to monitor
   :query timeframe: Time frame for analysis
   :statuscode 200: No error

.. http:get:: /api/resources/usage-patterns

   Get resource usage patterns.

   :query resource_type: Type of resource
   :query start_time: Start time for analysis
   :query end_time: End time for analysis
   :statuscode 200: No error

.. http:get:: /api/resources/bottlenecks

   Get resource bottleneck identification.

   :query analysis_period: Period for analysis (default: '24h')
   :query threshold: Bottleneck threshold (default: 0.8)
   :statuscode 200: No error

System Status
------------

.. http:get:: /api/system/status

   Get overall system status.

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
          "agents": {
              "content": {"status": "healthy"},
              "seo": {"status": "healthy"},
              "keyword": {"status": "healthy"},
              "crawler": {"status": "healthy"}
          },
          "tasks": 10,
          "active_tasks": 2,
          "system_health": "optimal",
          "last_update": "2024-02-04T12:00:00Z"
      }

Error Responses
--------------

.. http:any:: /api/*

   Common error responses for all endpoints.

   **Example error response**:

   .. sourcecode:: http

      HTTP/1.1 400 Bad Request
      Content-Type: application/json

      {
          "error": "Invalid parameters",
          "details": "Missing required field: type"
      }

   :statuscode 400: Bad request
   :statuscode 401: Authentication required
   :statuscode 403: Forbidden
   :statuscode 404: Resource not found
   :statuscode 429: Too many requests
   :statuscode 500: Server error 