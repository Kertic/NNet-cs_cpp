using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NNet
{
    public class Layer
    {
        public Neuron[] NeuronArray;
        public Layer BackLayer;

        public Layer(Neuron[] neuronArray, Layer backLayer = null)
        {
            NeuronArray = neuronArray;
            for (int i = 0; i < neuronArray.Length; i++)
            {
                if (NeuronArray[i] == null)
                    NeuronArray[i] = new Neuron();
            }
        }

        public Vector<double> GetNeuronValues()
        {
            Vector<double> returnVector = Vector<double>.Build.Dense(NeuronArray.Length);
            for (int i = 0; i < NeuronArray.Length; i++)
            {
                returnVector[i] = NeuronArray[i].Activation;
            }

            return returnVector;
        }

        public string PrintActivations()
        {
            string returnString = "{";

            for (int i = 0; i < NeuronArray.Length - 1; i++)
            {
                returnString += NeuronArray[i].Activation + ",";
            }

            returnString += NeuronArray[NeuronArray.Length - 1].Activation;
            returnString += "}";
            return returnString;
        }

        public Vector<double> GetNeuronInfluences()
        {
            List<double> neurons = new List<double>();
            for (int i = 0; i < NeuronArray.Length; i++)
            {
                var vec = NeuronArray[i].GetInfluencesWeights().ToArray();
                for (int j = 0; j < vec.Length; j++)
                    neurons.Add(vec[j]);
            }

            Vector<double> returnVec = Vector<double>.Build.DenseOfArray(neurons.ToArray());

            return returnVec;
        }
    }
}