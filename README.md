![image](https://github.com/imcjp/Opacus-DPCR/blob/main/data/img/logo.png)

Opacus-DPCR is a library extend the widely used [**Opacus**](https://github.com/pytorch/opacus) for learning with differential privacy (DP). Compared with Opacus, our Opacus-DPCR introduces differential privacy continuous data release (DPCR) to improve learning accuracy. It integrates various DPCR models (including our proposed BCRG and ABCRG). For more details, please refer to our work "A Federated Learning Framework Based on Differential Privacy Continuous Data Release".

In Opacus-DPCR, we support multiple DPCR models for building parametric models. They are as follows:
1. SimpleMech: Essentially the same as Opacus. They both add noise after each gradient calculation and then accumulate the noisy gradients to construct the parametric models.
2. TwoLevel: The method proposed in [1]. It is a primary DPCR model.
3. BinMech: The method proposed in [1]. It is a classical DPCR model, applying a binary tree to achieve a logarithmically growing RMSE.
4. FDA: The method proposed in [1]. It first applies binary indexed tree (BIT) on DPCR for higher accuracy. But it is only suitable for Laplacian noise.
5. BCRG: The method proposed in our work. It designs the optimization method for BIT-based DPCR with Gaussian noise.
6. ABCRG: The method proposed in our work. It improves the BCRG by using the residual sensitivity to further boost the accuracy.

Our Opacus-DPCR keeps a high compatibility with Opacus. You only need to easily modify a few lines of code for applying our achievements.

For example, in a classic Opacus example from (https://github.com/pytorch/opacus/blob/main/examples/mnist.py), you just do the following modifications to apply Opacus-DPCR. For more details, see our [**Example Code**](https://github.com/imcjp/Opacus-DPCR/blob/main/demo/opacusDpcrTest.py).
