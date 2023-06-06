#import "template.typ": *


#show: project.with(
  title: "A Literature Summary: \nReduce Clever-Hans effects in Neural Networks",
  authors: (
    "Jim Neuendorf\nExplainable AI for Decision Making, SS 2023",
  ),
  abstract: [
    With the rise of popularity of artificial intelligence (AI) in daily applications,
    the need for the explanation of a model's prediction has also become more important.
    Some of the current research focuses on neural networks (NN) that produce the correct output for the wrong reasons. This effect is called the Clever Hans (CH).
    The papers summarized in this work contribute methods related to explanation for reducing these effects and thus making models more robust.
  ],
)



= Introduction

AI is used in all kinds of applications across many domains. In certain situations, for example medical assistance, decision errors are intolerable.
Since AI models typically do not achieve 100% accuracy on complex tasks, explainable AI (XAI) is required to establish trust, fairness and transparency by providing answers to _wh_-questions, e.g. "#strong[Wh]y did the model predict something?"
Sometimes, explainability is even mandatory by law such as in the European Union with its _right to explanation_.
Consequently, XAI techniques must applicable to models processing all imaginable types of data, i.e. images, audio, video, text etc. @xai-status.
Furthermore, foundation models are becoming popular so fighting CH effects is a timely concern @preemptive-prune-ch.



= Overview <overview>

In order to detect CH effects, XAI can be a helpful tool. Therefore, we first look at "methods to make explanations [...] more robust against attacks that manipulate the input" @towards-robust-ex.
We then discuss methods for reducing CH effects using XAI that either (1) retrain the model using modified data or an adjusted objective or (2) change the model structurally, e.g. by inserting layers. The latter is most common among the following and called _post-hoc_ because the model regarded is already trained @xai-status.

// When the output of a model is supposed to be changed, we can (1) modify the training data and retrain the model or (2) change the model's structure without retraining it, e.g. by inserting layers.

//Most of the following approaches to CH-reduction fall into the latter category which is called _post-hoc_ because the model in question is already trained. In the next sections we look at different methods from four papers followed by an evaluation questioning the effectiveness of post-hoc explanations.

//XAI techniques can be classified into two categories: (1) post-hoc and (2) transparent methods @xai-status. Because in this work we focus on neural networks (NN) and "transparent methods are such methods where the inner working and decision-making [...] is simple to interpret and represent" @xai-status, the rest of this summary considers _post-hoc_ explanation methods.



= Robust Explanations

As mentioned in @overview explanations can be made more robust using three relatively simple techniques. These are derived from the finding that the maximum possible explanation change is bounded by the Frobenius norm of the Hessian $norm(H)_F$ @towards-robust-ex. Hence, the presented approaches try to minimize $norm(H)_F$ -- within the context of supervised learning.


== Curvature Minimization

An intuitive solution is to penalize the F-Norm by adding a term to the loss function.
Because "calculating the Frobenius norm of the Hessian es expensive, [...] [they] propose to estimate the F-norm stochastically" @towards-robust-ex with an expectation value using Monte-Carlo sampling.

Note that the impact of the penalization term can be controlled by a hyperparameter.


== Weight Decay

Because $norm(H)_F$ also depends on the weights of the NN, weight decay can also be used to robustify the explanation.
This method is a regularization aiming for small weight values. In this case using the Frobenius norm -- other regularizations such as $L^2$ could work as well.
Weight decay is a well-known technique for improving model generalization, but in the context of XAI it is new.

It is notable that a technique that improves model generalization simultaneously helps enhancing a model's explanation robustness.


== Smooth Activation Functions

As a third approach activation functions with smaller maximum values of the first and second derivatives result in smaller values for $norm(H)_F$.
The Softplus non-linearity is used as an example for a smooth function. However, effects on training of the model are not presented (e.g. in constrast to ReLU).


== Results

The three presented techniques could be validated independently using a convolutional neural network (CNN) on the CIFAR-10 dataset.
Even a small reduction of the curvature led to a significant improvement of the explanation robustness. However, the model accuracy decreases the more the robustness increases. Thus, there is a trade-off between the two effects.



= Reducing Clever-Hans Effects

This section contains attempts to reduce CH effects in both supervised and unsupervised settings using post-hoc as well as fine-tuning approaches.



== Bagging in Anomaly Detection

Research on "whether Clever Hans also occurs in unsupervised learning, and in which form, has received so far alsmost no attention." @ch-anomaly-detection
So far only model-specific approaches have been proposed, i.e. no general technique that applies to all anomaly detection models.
Anomaly detection approaches can be categorized into _density-_, _reconstruction-_, and _boundary-based_.
// Note that boundary-based models are somewhat special as they can use supervised data to form their boundary (supervised learning).

Therefore, the researchers "introduce a common XAI framework that is applicable to a broad range of anomaly detection models" @ch-anomaly-detection: They use a NN architecture with three layers for explanation extraction using Deep Tayler Decomposition (a mechanism similar to Layer-wise Relevance Propagation (LRP)). For each of the three layers _feature extraction_, _distance_, and _pooling_ they define (1) specific calculations and (2) certain propagation rules -- both depending on the type of model#footnote()[The propagation rules (backward pass) are model-dependent because the calculation (forward pass) also is.].

For example, the distance (forward pass) of a reconstruction-based model is the Mean Squared Error (MSE), its "outlier score", $o(x) = ||r(x) - x||^2$. For the attribution (backward pass) in the distance layer the mean $mu_(j k)$ is actually used as opposed to being a constant for the other model types.


=== Evaluation

Using the datasets MNIST-C and MVTec, both coming with anomaly-ground-truth data, and a Clever-Hans score#footnote()[The score measures the mismatch between the detection and explanation accuracy.], they find that the different "models are affected by the problem in different ways" @ch-anomaly-detection.
// These high outlier scores, even for classes with high accuracy, were produced for the wrong reasons, i.e. due to the CH effect.
Thus, they "hypothesize that [CH effects] are inherent to the structre of the anomaly detection models rather than [...] the trainig data." @ch-anomaly-detection
This hypothesis is followed by a reasoning about why this is the case.

Sticking with the autoencoder example, they argue that the CH effect is caused by samples whose reconstruction is far away from the input distribution. Thus, an outlier is detected due to features that do not relate to the input distribution which means the outlier detection is unrelated to the actual anomaly.


=== Solution

Because the source of CH is not in the data but in in the models themselves, the idea is to allow "multiple anomaly models to mutually cancel their individual structural weaknesses" @ch-anomaly-detection.
Thus, a bagging approach of outlier scores is proposed:
$ o_"Bag"(x) = 1/3 lr((o_"KDE"(x) + o_"Auto"(x) + o_"Deep"(x)), size: #180%) $
where KDE is density-, Auto is boundary-, and Deep is boundary-based.

"The bagged model ranks first among all four models [...] although relatively far from the ground-truth" @ch-anomaly-detection. So there is still a lot of room for improvement with techniques going "beyond simple bagging [...] and structurally less rigid models" @ch-anomaly-detection.



== Pruning in Deep Models

CH effects can remain undetected even if the user's explanation agrees with the one from XAI.
In this case, there is "neither prior knowledge about the spurious feature, nor data containing it" @preemptive-prune-ch:
We receive the trained model which "is to be robustified post-hoc with limited data" @preemptive-prune-ch without CH features, apply the soft-pruning, and should be able to deploy the model.

The proposed pruning method is called _Explanation-Guided Exposure Minimization_ (EGEM). They arrive at a practical formulation which comprises (1) explanations of the refined model to be close to those of the original one and (2) a penalty used for exposure minimization.
The pruning strength depends on each neuron's activation frequency and magnitude.

A second pruning approach _PCA-EGEM_ is introduced for more effective pruning: In PCA space the features are more disentangled so pruning should be more effective.


=== Evaluation

In their work the scientists could validate approve the effectiveness of the pruning method using datasets were the spurious artifacts were known:
MNIST with manual poisoning, and ImageNet and ISIC were previous work has identified (potential) CH correlations.

Afterwards, the approach was tested in an exploratory manner on the CelebA dataset. Using PCA-EGEM it could be observed that "a model bias [...] has been mitigated" and it "enables the retrieval of a more diverse set of positive instances from a large heterogeneous dataset" @preemptive-prune-ch.



== Class Artifact Compensation in Deep Models

Another unlearning approach using XAI called _Class Artifact Compensation_ (ClArC) is provided by @find-remove-ch.
This work focuses on automating the process of CH removal because with datasets getting larger, manual inspection/curation becomes infeasible.
//For doing so a three-step pipeline is proposed: (1) identification, (2) description, and (3) suppression of CH effects.


=== Large-scale Analysis

For semi-automated discovery a technique called _Spectral Relevance Analysis_ (SpRAy) is used which aims to bridge local and global XAI by inspecting large sets of local explanations. From attribution maps it computes a spectrum of eigenvalues via spectral clusters. The eigenvalue spectrum can be used "for ranking [...] analyzed classes w.r.t. their potential for exhibiting CH phenomena" @find-remove-ch.

In addition to enhancing SpRAy's cluster visualization and labeling by making use of an intermediate result, they propose using Fisher Discriminant Analysis
#footnote()[
  FDA maximizes between-class and minimizes within-class scatter.
  Other separation algorithms possible.
] (FDA) to rank class-wise clusterings by their separability $tau$, especially when there are many classes resulting in many clusters @find-remove-ch. Large values of $tau$ indicate artifiact candidates.


=== Unlearning with Augmentative Class Artifact Compensation

Here (A-ClArC), the training data is augmented in a way that the trained classifier becomes insensitive to an artifact.
This is achieved by adding an artifact found with SpRAy to some samples of all other classes using a _forward artifact model_.

When the classifier is trained with the augmented data, it can no longer rely on the artifact as an "easy shortcut" for classification but must learn other strategies.


=== Unlearning with Projective Class Artifact Compensation

In constrast to the before-mentioned method that requires retraining the model, the second proposed approach (P-ClArC) suppresses CH artifacts
#footnote()[It does not perform true unlearning in the same way as A-ClArC.].
This happens "without retraining by incorporating a _backward artifact model_ [...] directly into the prediction model" @find-remove-ch.
While the idea is basically the same as for A-ClArC, it works by projecting the data points to a position of the decision boundary to which the estimated articat direction is normal.
Therefore, the boundary ignores and thus suppresses the artifact while leaving the output unchanged for clean samples.


=== Evaluation

The paper contains a very thorough evaluation of the extended SpRAy and ClArC methods on six datasets.
What they found is that their extended SpRAy (1) picks up most artifacts when there are any, even though their importance varies between models and classes, and (2) still requires human judgement for the final decision on CH candidates.

Concerning their proposed methods they conclude that both A-ClArC and P-ClArC perform very well -- in input as well as latent space.
Two mentioned limitations are that A-ClArC is time-consuming and "P-ClArC will not lead to an increased generalization performance, since the model never has a chance to adapt its weights [...] and correct its faulty prediction" @find-remove-ch.

Hence, common datasets like ImageNet can be "un-Hansed" to provide a more unbiased basis for foundation models.



== Post-hoc Explanations for Unknown Spurious Correlation

The post-hoc approaches discussed so far all relied on XAI and seemed very promising.
In the last paper of this summary, the "post-hoc explanation methods tested are ineffective when the spurious artifact is unknwon at test-time" @post-hoc-ex.
The used explanations methods are (1) feature attribution, (2) concept activation, and (3) training point ranking.

In order to compare these methods, they define a _spurious score_
#footnote()[
  In other words, it "is the probability that the model assigns the input to the spurious aligned class if the spurious signal is added to the input" @towards-robust-ex.
]
which quantifies "the strength of a model's dependence on a training signal" @towards-robust-ex.

For their evaluation, they use three measures comparing different combinations of explanations
#footnote()[
  (1) 
]
with three similarity functions
#footnote()[
  One for each attribution method.
].
Their findings range from (1) struggling "to detect blur signals even when known" over (2) "concept rankings can help detect reliance on the visible signals but not non-visible" to (3) "all methods struggle to reliably indicate that spurious models are reliant on the blur signal" @towards-robust-ex.



= Summary

With the last paper dulling the results outlined previously, it will be interesting to see how the big picture evolves, e.g. how P-ClArC performs across different models and tasks when evaluated in the spurious-score settings. Additionally, further research on how the structure of a model affects explanations and their robustness in supervised learning could impact the development of future foundation models
#footnote()[
  For example, DINOv2 has amazing emergent properties concerning the latent-space structure.
].

/* 
#pagebreak()

#heading(numbering: none)[Terms]

/ AI: Artificial intelligence
/ CH: Clever Hans
/ ML: Machine learning
/ NN: Neural network
/ XAI: Explainable AI
/ LRP : Layer-wise relevance propagation
/ MSE: Mean squared error
/ DNN: Deep neural network
/ CNN: Convolutional neural network
/ ClArC: Class artifact compensation
/ EGEM: Explanation-guided exposure minimization
/ SpRAy: Spectral relevance analysis
/ FDA: Fisher Discriminant Analysis
*/
 
#bibliography("refs.yml", title: "References")
