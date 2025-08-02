from docling.grpclient.inference_client import InferenceClient
from docling.proto.inference_pb2 import (
    PromptRequest,
    CompletionRequest,
    CompletionResponse,
    ModelParams,
    ModelTrait,
    SamplerParams,
    UserMessage,
    UserContent,
    Message,
    AssistantMessage,
    AssistantContent,
    Image,
    ImageMediaType
)
import asyncio

async def main():
    try:
        path = "/Users/chenxin/Downloads/20250716-170704.jpeg"
        with open(path, "rb") as f:
            img_bytes = f.read()

        client = InferenceClient(
            "127.0.0.1:9988",
            "/Users/chenxin/projects/powermers/k8s-scripts/local/certs/agentic/req-barm-server/ca.pem",
            "/Users/chenxin/projects/powermers/k8s-scripts/local/certs/agentic/req-barm-server/client_cert.pem",
            "/Users/chenxin/projects/powermers/k8s-scripts/local/certs/agentic/req-barm-server/client_key.pem",
        )
        await client.connect()

        request = CompletionRequest(
            model_params=ModelParams(
                model_trait=ModelTrait.MODEL_TRAIT_IMAGE_TO_TEXT,
                preamble="you are a helpful assistant",
                sampler_params=SamplerParams(
                    max_tokens=10000,
                    temperature=1.0,
                )
            ),
            user_message=UserMessage(
                user_contents=[
                    UserContent(
                        text="Convert this image to markdown. Do not miss any text and only output the bare markdown!"
                    ),
                    UserContent(
                        image=Image(
                            media_type=ImageMediaType.IMAGE_MEDIA_TYPE_JPEG,
                            data=img_bytes
                        )
                    )
                ]
            ),
        )

        response = await client.completion(request)
        print(f"Message: {response.message}")
        
        # print("\n--- Streaming Response ---")
        # items: list[str] = []
        # async for stream_response in client.stream_completion(request):
        #     items.append(stream_response.message)
        #     print(f"Stream chunk: {stream_response.message}")
        
        # full_message = "".join(items)
        # print(f"\n\nFull message: \n\n{full_message}")
        await client.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())