import asyncio
import base64
import grpc
from typing import Optional

# according to inference.proto generated Python file 
# generate command: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/inference.proto
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
from docling.proto.inference_pb2_grpc import InferenceStub

class InferenceClient:
    # _instance = None

    # def __new__(cls, address: str = "127.0.0.1:9988", ca_cert_path: Optional[str] = None, 
    #             client_cert_path: Optional[str] = None, client_key_path: Optional[str] = None):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls)
    #         cls._instance._initialized = False
    #     return cls._instance
    
    def __init__(self, address: str = "127.0.0.1:9988", ca_cert_path: Optional[str] = None,
                 client_cert_path: Optional[str] = None, client_key_path: Optional[str] = None):
        # if self._initialized:
        #     return
        self.address = address
        self.ca_cert_path = ca_cert_path
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[InferenceStub] = None
        # self._initialized = True
    
    def _load_credentials(self):
        """load mTLS credentials"""
        try:
            ca_cert = None
            if self.ca_cert_path:
                with open(self.ca_cert_path, 'rb') as f:
                    ca_cert = f.read()
            
            client_cert = None
            client_key = None
            if self.client_cert_path and self.client_key_path:
                with open(self.client_cert_path, 'rb') as f:
                    client_cert = f.read()
                with open(self.client_key_path, 'rb') as f:
                    client_key = f.read()
            
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert
            )
            
            return credentials
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Certificate file not found: {e}")
        except Exception as e:
            raise Exception(f"Failed to load mTLS credentials: {e}")
    
    async def connect(self):
        """connect to server"""
        if self.channel is None:
            if self.ca_cert_path or self.client_cert_path or self.client_key_path:
                credentials = self._load_credentials()
                self.channel = grpc.aio.secure_channel(
                    self.address, 
                    credentials,
                )
            else:
                self.channel = grpc.aio.insecure_channel(self.address)
            
            self.stub = InferenceStub(self.channel)
        return self
    
    async def prompt(self, request: PromptRequest) -> CompletionResponse:
        """send prompt request"""
        if not self.stub:
            await self.connect()
        response = await self.stub.Prompt(request)
        return response
    
    async def stream_prompt(self, request: PromptRequest):
        """send stream prompt request"""
        if not self.stub:
            await self.connect()
        async for response in self.stub.StreamPrompt(request):
            yield response
    
    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """send completion request"""
        if not self.stub:
            await self.connect()
        response = await self.stub.Completion(request)
        return response
    
    async def stream_completion(self, request: CompletionRequest):
        """send stream completion request"""
        if not self.stub:
            await self.connect()
        async for response in self.stub.StreamCompletion(request):
            yield response
    
    async def close(self):
        """close connection"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None