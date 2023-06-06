#import "template.typ": *


#show: project.with(
  title: "A Literature Summary: \nReduce Clever-Hans Effects in Neural Networks",
  authors: (
    "Jim Neuendorf\nExplainable AI for Decision Making, SS 2023",
  ),
  abstract: [
    As artificial intelligence (AI) is becoming more important, so is the need for the explanation of a model's prediction.
    Some of the current research focuses on neural networks (NN) that produce the correct output for the wrong reasons.
    That effect is called the Clever Hans (CH).
    We discuss methods related to explainable AI for reducing these effects and thus making models more robust.
  ],
)



= Introduction

In some of the many domains AI is used in, for example medical assistance, decision errors are intolerable.
Since AI models typically do not achieve 100% accuracy on complex tasks, explainable AI (XAI) is required to establish trust, fairness and transparency by providing answers to _wh_-questions, e.g. "#strong[Wh]y did the model predict something?"
Sometimes, explainability is mandatory by law such as in the European Union with its _right to explanation_.
Consequently, XAI techniques must applicable to models processing different types of data such as images, text @xai-status.
Furthermore, foundation models are becoming popular so reducing CH effects is a timely concern @preemptive-prune-ch.



= Overview

In order to detect CH effects XAI can be a helpful tool. Therefore, we first look at "methods to make explanations [...] more robust" @towards-robust-ex.
We then discuss methods for reducing CH effects using XAI that either (1) retrain the model using modified data and/or adjusted objectives or (2) change the model structurally, e.g. by inserting layers. The latter is called _post-hoc_ because the model is already trained @xai-status.

All papers except the one on anomaly detection focus on supervised learning.


= Robust Explanations

Explanations can be made more robust using three simple techniques.
These are derived from the finding that the maximum possible explanation change is bounded by the Frobenius norm of the Hessian $norm(H)_F$ @towards-robust-ex which the following minimize.


== Curvature Minimization

Here, we penalize $norm(H)_F$ by adding a term to the loss function.
Because "calculating the Frobenius norm of the Hessian es expensive, [...] [they] propose to estimate the Frobenius norm stochastically" @towards-robust-ex with an expectation value using Monte-Carlo sampling.

//Note that the impact of the penalization term can be controlled by a hyperparameter.


== Weight Decay

Because $norm(H)_F$ also depends on the weights of the NN, weight decay can be used to robustify the explanation.
This method is a regularization aiming for small weight values. In this case using the Frobenius norm -- other regularizations such as $L^2$ could work as well.
Weight decay is a technique for improving model generalization, but in the context of XAI it is new.

// It is notable that a technique that improves model generalization simultaneously helps enhancing a model's explanation robustness.


== Smooth Activation Functions

Another approach is using activation functions with smaller maximum values of the first and second derivatives.
This means the functions is _smooth_ which leads to smaller values for $norm(H)_F$.
Softplus is such a function.
Effects on the model performance are not presented (e.g. in constrast to ReLU).


== Results

The three presented techniques are validated independently using a convolutional neural network (CNN) on the CIFAR-10 dataset.
Even small curvature minimizations lead to a significant improvement of the explanation robustness.
However, the model accuracy decreases with better the robustness -- there is a trade-off.



= Reducing Clever-Hans Effects

// This section contains attempts to reduce CH effects in supervised and unsupervised settings using post-hoc as well as fine-tuning approaches.



== Bagging in Anomaly Detection

Research on "whether Clever Hans also occurs in unsupervised learning, and in which form, has received so far alsmost no attention." @ch-anomaly-detection
So far only model-specific approaches have been proposed, i.e. no general technique that applies to all anomaly detection models.
Anomaly detection models can be categorized into _density-_, _reconstruction-_, and _boundary-based_.
// Note that boundary-based models are somewhat special as they can use supervised data to form their boundary (supervised learning).

The researchers "introduce a common XAI framework that is applicable to a broad range of anomaly detection models" @ch-anomaly-detection:
They use a three-layer NN architecture for explanation extraction using Deep Tayler Decomposition (a mechanism similar to Layer-wise Relevance Propagation (LRP)).
For each of the three layers _feature extraction_, _distance_, and _pooling_ they define (1) specific calculations and (2) certain propagation rules -- both depending on the model type #footnote()[
  The propagation rules (backward pass) are model-dependent because the calculation (forward pass) also is.
].

For example, the distance (forward pass) of a reconstruction-based model is the Mean Squared Error (MSE), its "outlier score", $o(x) = ||r(x) - x||^2$.
For the attribution (backward pass) in the distance layer the mean $mu_(j k)$ is actually used as opposed to being constant for the other model types.


=== Evaluation

Using two datasets, both coming with anomaly-ground-truth data, and a Clever-Hans score#footnote()[
  The score measures the mismatch between the detection and explanation accuracy.
], they find that the different "models are affected by the problem in different ways" @ch-anomaly-detection.
// These high outlier scores, even for classes with high accuracy, were produced for the wrong reasons, i.e. due to the CH effect.
Thus, CH effects "are inherent to the structre of the anomaly detection models rather than [...] the trainig data" @ch-anomaly-detection.
This hypothesis is followed by a reasoning about why this is the case.

Sticking with the autoencoder example, they argue that the CH effect is caused by samples whose reconstruction is far away from the input distribution. Thus, an outlier is detected due to features that do not relate to the input distribution which means the outlier detection is unrelated to the actual anomaly.


=== Solution

Because the source of CH is not in the data but in in the models themselves, the idea is to allow "multiple anomaly models to mutually cancel their individual structural weaknesses" @ch-anomaly-detection.
They propose a bagging approach of outlier scores#footnote()[
  In the equation KDE is a density-, Auto a boundary-, and Deep a boundary-based model.
]:
$ o_"Bag"(x) = 1/3 lr((o_"KDE"(x) + o_"Auto"(x) + o_"Deep"(x)), size: #180%) $

"The bagged model ranks first among all four models [...] although relatively far from the ground-truth" @ch-anomaly-detection. There is still room for improvement by going "beyond simple bagging [...] and structurally less rigid models" @ch-anomaly-detection.



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
This work focuses on automating the process of CH removal because with datasets getting larger, manual inspection becomes infeasible.
//For doing so a three-step pipeline is proposed: (1) identification, (2) description, and (3) suppression of CH effects.


=== Large-scale Analysis

For semi-automated discovery they use _Spectral Relevance Analysis_ (SpRAy) which aims to bridge the gap between local and global XAI by inspecting large sets of local explanations.
From attribution maps it computes a spectrum of eigenvalues via spectral clusters. That spectrum can be used "for ranking [...] analyzed classes w.r.t. their potential for exhibiting CH phenomena" @find-remove-ch.

They enhance SpRAy's cluster visualization and labeling by using an intermediate result and propose using Fisher Discriminant Analysis
#footnote()[
  FDA maximizes between-class and minimizes within-class scatter.
  Other separation algorithms are possible.
] (FDA) to rank class-wise clusterings by CH likeliness, especially with many-class datasets @find-remove-ch.
// Large values of $tau$ indicate artifiact candidates.


=== Unlearning with Augmentative Class Artifact Compensation

Here (A-ClArC), the training data is augmented so that the trained classifier becomes insensitive to an artifact.
This is achieved by adding an artifact found with SpRAy to some samples of all other classes using a _forward artifact model_.

When the classifier is trained with the augmented data, it can no longer rely on the artifact as an "easy shortcut" for classification and must learn "real" strategies.


=== Unlearning with Projective Class Artifact Compensation

The second approach (P-ClArC) suppresses CH artifacts/* #footnote()[
  It does not perform true unlearning in the same way as A-ClArC.
]. */
"without retraining by incorporating a _backward artifact model_ [...] directly into the prediction model" @find-remove-ch.
While the idea is the same as for A-ClArC, data points are projected to a position of the decision boundary to which the estimated artifact direction is normal.
The boundary ignores the artifact while leaving clean samples' output unchanged.


=== Evaluation

The paper contains a very thorough evaluation of the extended SpRAy and ClArC methods on six datasets.
They find that their extended SpRAy (1) picks up most artifacts when there are any, even though their importance varies between models and classes, but (2) still requires human judgement for the final decision on CH candidates.

Concerning their proposed methods they conclude that both A-ClArC and P-ClArC perform very well -- both in input and latent space.
Two limitations are that A-ClArC is time-consuming and "P-ClArC will not lead to an increased generalization performance, since the model never has a chance to adapt its weights [...] and correct its faulty prediction" @find-remove-ch.

This way, common large datasets like ImageNet can be "un-Hansed" to provide a more unbiased basis for foundation models.



== Post-hoc Explanations for Unknown Spurious Correlation

The post-hoc approaches discussed so far all relied on XAI and seemed very promising.
In the last paper of this summary, the "post-hoc explanation methods tested are ineffective when the spurious artifact is unknwon at test-time" @post-hoc-ex.
The used explanations methods are (1) feature attribution, (2) concept activation, and (3) training point ranking.

In order to compare these methods, they define a _spurious score_
/* #footnote()[
  In other words, it "is the probability that the model assigns the input to the spurious aligned class if the spurious signal is added to the input" @towards-robust-ex.
] */
which quantifies "the strength of a model's dependence on a training signal" @towards-robust-ex.

They use three measures comparing different combinations of explanations
#footnote()[
  (1) Known spurious signal detection $S(E_f_(s p u)(x_(s p u)), x_(g t))$, (2) cause-for-concern $S(E_f_(s p u)(x_(n o r m)), E_f_(n o r m)(x_(n o r m)))$, and (3) false alarm $S(E_f_(n o r m)(x_(s p u)), E_f_(s p u)(x_(s p u)))$, where $S$: similarity, $E_f_(s p u)$: spurious-model expl., $E_f_(n o r m)$: normal-model expl., $x_(g t)$: ground-truth expl., $x_(n o r m)$: normal input, and $x_(s p u)$: spurious input.
]
with three similarity functions (one for each attribution method).
Their findings range from (1) struggling to detect signals even when known over (2) "can help detect reliance on the visible signals but not non-visible" to (3) "all methods struggle to reliably indicate that spurious models are reliant on the blur signal" @towards-robust-ex.



= Summary

With the last paper dulling the results outlined previously, it will be interesting to see how the big picture evolves, e.g. how P-ClArC performs across different models and tasks when evaluated in the spurious-score settings. Additionally, further research on how the structure of a model affects explanations and their robustness in supervised learning could impact the development of future foundation models
#footnote()[
  DINOv2 has amazing emergent properties concerning the latent-space structure.
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
