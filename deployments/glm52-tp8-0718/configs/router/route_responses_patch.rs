
    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream.unwrap_or(false);

        // Extract text for cache-aware routing from ResponsesRequest input items
        let request_text = if self.policies_need_request_text() {
            Self::build_responses_request_text(body)
        } else {
            None
        };

        // Use n=1 as default batch size since Responses API doesn't have explicit n param
        let batch_size = Some(1usize);

        let context = PDRequestContext {
            route: "/v1/responses",
            batch_size,
            is_stream,
            return_logprob: false,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

