#pragma warning disable SKEXP0070
#pragma warning disable SKEXP0050
#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0010

// Create a chat completion service
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Plugins.Memory;
using SmartComponents.LocalEmbeddings.SemanticKernel;
using Microsoft.ML.OnnxRuntime;
using feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI;

//------------------------------
using System;
using System.IO;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Microsoft.SemanticKernel.Data;
using Microsoft.SemanticKernel.PromptTemplates.Handlebars;
using local_rag_sk.Helpers;

internal class Program
{
    private static async Task Main(string[] args)
    {

        // Your PHI-3 model location 
        var phi3modelPath = @"D:\models\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32";
        var bgeModelPath = @"D:\models\bge-micro-v2\onnx\model.onnx";
        var vocabPath = @"D:\models\bge-micro-v2\vocab.txt";

        // Load the model and services
        var builder = Kernel.CreateBuilder();
        builder.AddOnnxRuntimeGenAIChatCompletion(phi3modelPath, "phi-3");
        builder.AddBertOnnxTextEmbeddingGeneration(bgeModelPath, vocabPath);

        // Build Kernel
        var kernel = builder.Build();

        // Create services such as chatCompletionService and embeddingGeneration
        var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
        var embeddingGenerator = kernel.GetRequiredService<ITextEmbeddingGenerationService>();

        // Setup a memory store and create a memory out of it
        var memoryStore = new VolatileMemoryStore();
        var memory = new SemanticTextMemory(memoryStore, embeddingGenerator);

        // Loading it for Save, Recall and other methods
        kernel.ImportPluginFromObject(new TextMemoryPlugin(memory));

        // Populate the memory with some interesting facts
        string collectionName = "TheLevelOrg";
        MemoryHelper.PopulateInterestingFacts(memory, collectionName);

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("""
                              _                     _   _____            _____ 
                             | |                   | | |  __ \     /\   / ____|
                             | |     ___   ___ __ _| | | |__) |   /  \ | |  __ 
                             | |    / _ \ / __/ _` | | |  _  /   / /\ \| | |_ |
                             | |___| (_) | (_| (_| | | | | \ \  / ____ \ |__| |
                             |______\___/ \___\__,_|_| |_|  \_\/_/    \_\_____|         
                                                               by Arafat Tehsin              
                            """);

        // Start the conversation
        while (true)
        {
            // Get user input
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("User > ");
            var question = Console.ReadLine()!;

            // Enable auto function calling
            OpenAIPromptExecutionSettings openAIPromptExecutionSettings = new()
            {
                ToolCallBehavior = ToolCallBehavior.EnableKernelFunctions,
                MaxTokens = 200
            };

            var response = kernel.InvokePromptStreamingAsync(
                promptTemplate: @"{{$input}}",
                arguments: new KernelArguments(openAIPromptExecutionSettings)
                {
            { "input", question }
                });

            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("\nAssistant > ");

            string combinedResponse = string.Empty;
            await foreach (var message in response)
            {
                //Write the response to the console
                Console.Write(message);
                combinedResponse += message;
            }

            Console.WriteLine();
        }
    }
}