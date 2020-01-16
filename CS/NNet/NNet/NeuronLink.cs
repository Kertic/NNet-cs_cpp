using System;
using System.Diagnostics.Contracts;
 
 namespace NNet
 {
     public class NeuronLink
     {
         public Neuron Back;
         public Neuron Front;
         public double Weight;

         public NeuronLink(Neuron back=null, Neuron front = null, double weight = 0.0f)
         {
             Back = back;
             Front = front;
             Weight = weight;
         }

         public double GetWeightedSum()
         {
             return Back.Activation * Weight;
         }
     }
 }