# Action-Quality-Assessment-on-Tennis-Strokes-executions-using-a-BiLSTM-based-Deep-Learning-Framework

How often does an amateur tennis player want to compare his own execution with that of the world’s greatest tennis players, such as Federer, Nadal or Djokovic? How often does a tennis school teacher watch his students hit the ball and give advice on how to improve their game based only on their eyesight? The **goal** we set with this project is to **automatically perform an action quality assessment on the execution of a tennis stroke, compared to the one of the experts in the field**. To achieve this, we developed a Deep Learning Model with a BiLSTM-based architecture. Starting from a video recording, it produces a score that evaluates how closely the movement approximates the one of a professional. In particular, the analysis of a stroke execution is focused on studying the evolution over time of the positions of different body parts, obtained through the pre-trained model OpenPose. This information has been processed in such a way that the **model is aware of both the importance of certain joints during the movement** (through the development of **Principal Component Analysis**) **and the temporal instants in which they have the greatest impact on the overall execution** (through **discrete-time analysis of the change in position of each joint**). The dataset used for training the model is **THETIS**: it contains videos of amateurs and experts for each tennis stroke. This subdivision is crucial for the acquisition of the reference scores, which are obtained by analyzing the performance of each execution against all the expert ones.

# OUR NEURAL NETWORK

We developed two versions of our model: 
- Architecture with **one BiLSTM**, joint evolution weights multiplied to the input skeletal data and the PCA inputs, concatenated to the output of the LayerNorm operation.
  ![image](https://github.com/user-attachments/assets/beafe7fa-dd88-47a7-9212-f82d6c092681)

- architecture with **two BiLSTMs**, joint evolution weights multiplied to the input skeletal data and the PCA inputs, concatenated to the output of the first dense layer.
  ![image](https://github.com/user-attachments/assets/d388508d-3d9c-4b94-90f2-f8c50d110cab)

# HIGHLIGHTS

- unprecedented solution for the Action Quality Assessment task over the Tennis domain, with capabilities to evaluate all the existing strokes (10 different types);
- computation of quality scores through statistical analysis, which ensures objectivity in evaluations, and creation of weights that emphasize the most relevant body parts during the execution of the stroke;
- developing of deep neural network models, composed by one or two BiLSTMs, capable of producing a quality score prediction given in input skeletal data;
- comparison with Liao et al.’s "SpatioTemporalNN" model applied on the THETIS dataset;
- BiLSTM with weights as the model with the best results, outdoing SpatioTemporalNN performance.
