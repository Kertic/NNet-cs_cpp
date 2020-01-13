using System;
using System.Diagnostics.Contracts;
 
 namespace NNet
 {
     public class NeuronLink
     {
         public Neuron Back;
         public Neuron Front;
         public float Weight;
         public float Bias;

         public NeuronLink(Neuron back=null, Neuron front = null, float weight = 0.0f, float bias = 0.0f)
         {
             Back = back;
             Front = front;
             Weight = weight;
             Bias = bias;
         }

         public double GetWeightedSum()
         {
             return Back.Activation * Weight + Bias;
         }
     }
 }