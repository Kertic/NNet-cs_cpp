using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NNet
{
    public class Layer
    {
        public Neuron[] NeuronArray;
        public Layer BackLayer;

        public Layer(Neuron[] neuronArray, Layer backLayer)
        {
            NeuronArray = neuronArray;
            for (int i = 0; i < neuronArray.Length; i++)
            {
                if(NeuronArray[i] == null)
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
    }
}