using System;
using System.Diagnostics.Contracts;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NNet
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            Layer test1 = new Layer(new Neuron[]
            {
                new Neuron(0.0f), 
                new Neuron(1.0f), 
            }, null);
            Layer test2 = new Layer(new Neuron[2], test1);
            Layer testOutput = new Layer(new Neuron[1], test2);

            Network testNet = new Network(new[]
            {
                test1, test2, testOutput
            });
            testNet.RandomizeWeightsAndBiases();
            testNet.WriteWeightsAndBiases("output1.txt");
            testNet.ReadInWeightsAndBiases("output2.txt");

            Layer test3 = new Layer(new[] {new Neuron(1.0f), }, test1);

            Console.WriteLine("Output layer: " +testOutput.PrintActivations());
            Console.WriteLine("Valid Layer: " + test3.PrintActivations());
            Console.WriteLine("Error: " + testNet.Cost(test3));
            Console.WriteLine("Network: \n" + testNet.PrintNetwork());
            Console.WriteLine("Network weights: \n" + testNet.GetWeights());
        }
    }
}