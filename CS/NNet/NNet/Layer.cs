using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NNet
{
    public class Layer
    {
        public Neuron[] NeuronArray;
        public Layer backLayer;

        public Layer(Neuron[] neuronArray, Layer backLayer)
        {
            NeuronArray = neuronArray;
            for (int i = 0; i < neuronArray.Length; i++)
            {
                if(NeuronArray[i] == null)
                    NeuronArray[i] = new Neuron();
            }
        }

        public Vector<float> GetNeuronValues()
        {
            Vector<float> returnVector = new DenseVector(NeuronArray.Length);
            for (int i = 0; i < NeuronArray.Length; i++)
            {
                returnVector[i] = NeuronArray[i].Value;
            }

            return returnVector;
        }
    }
}