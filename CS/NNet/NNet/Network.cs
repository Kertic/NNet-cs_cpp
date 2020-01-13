using System;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

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
                Layers[i].backLayer = Layers[i - 1];
                for (int j = 0; j < Layers[i].NeuronArray.Length; j++)
                {
                    Neuron currentNeuron = Layers[i].NeuronArray[j];
                    currentNeuron.LinksFromPreviousLayer = new NeuronLink[Layers[i - 1].NeuronArray.Length];
                    for (int k = 0; k < currentNeuron.LinksFromPreviousLayer.Length; k++)
                    {
                        currentNeuron.LinksFromPreviousLayer[k] = new NeuronLink();
                    }
                }
            }
        }

        public void RandomizeWeightsAndBiases()
        {
            Random rand = new Random();
            for (int i = 1; i < Layers.Length; i++)
            {
                for (int backLayerNeuronIndex = 0;
                    backLayerNeuronIndex < Layers[i].backLayer.NeuronArray.Length;
                    backLayerNeuronIndex++)
                {
                    Neuron backNeuron = Layers[i].backLayer.NeuronArray[backLayerNeuronIndex];
                    for (int currentLayerNeuronIndex = 0;
                        currentLayerNeuronIndex < Layers[i].NeuronArray.Length;
                        currentLayerNeuronIndex++)
                    {
                        Neuron frontNeuron = Layers[i].NeuronArray[currentLayerNeuronIndex];
                        frontNeuron.LinksFromPreviousLayer[backLayerNeuronIndex].Back = backNeuron;
                        frontNeuron.LinksFromPreviousLayer[backLayerNeuronIndex].Front = frontNeuron;
                        frontNeuron.LinksFromPreviousLayer[backLayerNeuronIndex].Bias = (float) rand.NextDouble();
                        frontNeuron.LinksFromPreviousLayer[backLayerNeuronIndex].Weight = (float) rand.NextDouble();
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
                                     currentNeuron.LinksFromPreviousLayer.Length);
                    for (int k = 0; k < currentNeuron.LinksFromPreviousLayer.Length; k++)
                    {
                        NeuronLink link = currentNeuron.LinksFromPreviousLayer[k];
                        writer.WriteLine("Weight:" + link.Weight);
                        writer.WriteLine("Bias:" + link.Bias);
                    }
                }
            }

            writer.Close();
        }

        public void ReadInWeightsAndBiases(string filePath)
        {
            StreamReader reader = new StreamReader(filePath);
            string header = reader.ReadLine();
            int layerCount = Int32.Parse(header.Split(':').Last());
            if (layerCount != Layers.Length - 1)
                throw new Exception("Invalid Layer count of: " + layerCount +
                                    " doesn't match with network layer count of " + Layers.Length);
            for (int i = 1; i < layerCount+1; i++)
            {
                string LayerHeader = reader.ReadLine();
                int nodeCount = Int32.Parse(LayerHeader.Split(':').Last());
                if (nodeCount != Layers[i].NeuronArray.Length)
                    throw new Exception("Invalid neuron count of: " + nodeCount +
                                        " doesn't match with layer neuron count of " + Layers[i].NeuronArray.Length);
                for (int j = 0; j < nodeCount; j++)
                {
                    Neuron currentNeuron = Layers[i].NeuronArray[j];
                    string NodeHeader = reader.ReadLine();
                    int previousNodeCount = Int32.Parse(NodeHeader.Split(':').Last());
                    if (previousNodeCount != currentNeuron.LinksFromPreviousLayer.Length)
                        throw new Exception("Invalid neuron count of: " + previousNodeCount +
                                            " doesn't match with layer neuron count of " +
                                            currentNeuron.LinksFromPreviousLayer.Length);
                    for (int k = 0; k < previousNodeCount; k++)
                    {
                        string weightString = reader.ReadLine();
                        string biasString = reader.ReadLine();
                        float weight = float.Parse(weightString.Split(':').Last());
                        float bias = float.Parse(biasString.Split(':').Last());

                        currentNeuron.LinksFromPreviousLayer[k].Weight = weight;
                        currentNeuron.LinksFromPreviousLayer[k].Bias = bias;
                    }
                }
            }


            reader.Close();
        }

        public float Cost(Layer perfectLayer)
        {
            Vector<float> vec = Layers[Layers.Length - 1].GetNeuronValues();
            Vector<float> perfVec = perfectLayer.GetNeuronValues();
            int max = vec.Count;
            int perfectMax = perfVec.Count;
            if (perfectMax != max)
                throw new Exception("Test layer is not the same dimension as final layer of network");

            Vector<float> diffSquared = DenseVector.Create(vec.Count, 0.0f);

            vec.Map2(
                ((a, b) => (float) Math.Pow((a - b), 2)),
                perfVec,
                diffSquared);

            return diffSquared.Sum();
        }
    }
}