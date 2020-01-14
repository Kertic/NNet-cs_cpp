using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

// ReSharper disable PossibleNullReferenceException

namespace NNet
{
    public class Network
    {
        public Layer[] Layers;

        public Network(Layer[] layers)
        {
            Layers = layers;
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].BackLayer = Layers[i - 1];
                for (int j = 0; j < Layers[i].NeuronArray.Length; j++)
                {
                    Neuron currentNeuron = Layers[i].NeuronArray[j];
                    currentNeuron.Influences = new NeuronLink[Layers[i - 1].NeuronArray.Length];
                    for (int k = 0; k < currentNeuron.Influences.Length; k++)
                    {
                        currentNeuron.Influences[k] = new NeuronLink();
                    }
                }
            }
        }

        public double Cost(Layer correctLayer)
        {
            Vector<double> vec = Layers[Layers.Length - 1].GetNeuronValues();
            Vector<double> correctVec = correctLayer.GetNeuronValues();
            int max = vec.Count;
            int perfectMax = correctVec.Count;
            if (perfectMax != max)
                throw new Exception("Test layer is not the same dimension as final layer of network");

            Vector<double> diffSquared = Vector<double>.Build.Dense(max);

            vec.Map2(
                ((a, b) => Math.Pow((a - b), 2)),
                correctVec,
                diffSquared);

            return diffSquared.Sum();
        }

        public Vector<double> GetWeights()
        {
            List<double> layerWeights = new List<double>();
            for (int i = 1; i < Layers.Length; i++)
            {
                var weights = Layers[i].GetNeuronInfluences();
                for (int j = 0; j < weights.Count; j++)
                {
                    layerWeights.Add(weights[j]);
                }
            }

            return Vector<double>.Build.DenseOfArray(layerWeights.ToArray());
        }

        #region Utility

        public void RandomizeWeightsAndBiases()
        {
            Random rand = new Random();
            for (int i = 1; i < Layers.Length; i++)
            {
                for (int backLayerNeuronIndex = 0;
                    backLayerNeuronIndex < Layers[i].BackLayer.NeuronArray.Length;
                    backLayerNeuronIndex++)
                {
                    Neuron backNeuron = Layers[i].BackLayer.NeuronArray[backLayerNeuronIndex];
                    for (int currentLayerNeuronIndex = 0;
                        currentLayerNeuronIndex < Layers[i].NeuronArray.Length;
                        currentLayerNeuronIndex++)
                    {
                        Neuron frontNeuron = Layers[i].NeuronArray[currentLayerNeuronIndex];
                        frontNeuron.Influences[backLayerNeuronIndex].Back = backNeuron;
                        frontNeuron.Influences[backLayerNeuronIndex].Front = frontNeuron;
                        frontNeuron.Influences[backLayerNeuronIndex].Weight = (float) rand.NextDouble();
                    }
                }
            }
        }

        public void WriteWeightsAndBiases(string filePath)
        {
            StreamWriter writer = new StreamWriter(filePath);
            writer.WriteLine("Layers past the first:" + (Layers.Length - 1));
            for (int i = 1; i < Layers.Length; i++)
            {
                writer.WriteLine("Number of Neurons in layer " + i + ":" + Layers[i].NeuronArray.Length);
                for (int j = 0; j < Layers[i].NeuronArray.Length; j++)
                {
                    Neuron currentNeuron = Layers[i].NeuronArray[j];
                    writer.WriteLine("Number of Neurons in reverse layer:" +
                                     currentNeuron.Influences.Length);
                    for (int k = 0; k < currentNeuron.Influences.Length; k++)
                    {
                        NeuronLink link = currentNeuron.Influences[k];
                        writer.WriteLine("Weight:" + link.Weight);
                    }
                }
            }

            writer.Close();
        }

        public void ReadInWeightsAndBiases(string filePath)
        {
            StreamReader reader = new StreamReader(filePath);
            string header = reader.ReadLine();
            int layerCount = int.Parse(header.Split(':').Last());
            if (layerCount != Layers.Length - 1)
                throw new Exception("Invalid Layer count of: " + layerCount +
                                    " doesn't match with network layer count of " + Layers.Length);
            for (int i = 1; i < layerCount + 1; i++)
            {
                string layerHeader = reader.ReadLine();
                int nodeCount = int.Parse(layerHeader.Split(':').Last());
                if (nodeCount != Layers[i].NeuronArray.Length)
                    throw new Exception("Invalid neuron count of: " + nodeCount +
                                        " doesn't match with layer neuron count of " + Layers[i].NeuronArray.Length);
                for (int j = 0; j < nodeCount; j++)
                {
                    Neuron currentNeuron = Layers[i].NeuronArray[j];
                    string nodeHeader = reader.ReadLine();
                    int previousNodeCount = int.Parse(nodeHeader.Split(':').Last());
                    if (previousNodeCount != currentNeuron.Influences.Length)
                        throw new Exception("Invalid neuron count of: " + previousNodeCount +
                                            " doesn't match with layer neuron count of " +
                                            currentNeuron.Influences.Length);
                    for (int k = 0; k < previousNodeCount; k++)
                    {
                        string weightString = reader.ReadLine();
                        float weight = float.Parse(weightString.Split(':').Last());

                        currentNeuron.Influences[k].Weight = weight;
                    }
                }
            }


            reader.Close();
        }

        public string PrintNetwork()
        {
            string returnString = "[\n";

            for (int i = 0; i < Layers.Length; i++)
            {
                string layer = Layers[i].PrintActivations();
                returnString += layer + "\n";
            }

            returnString += "]";
            return returnString;
        }
        
        #endregion
    }
}