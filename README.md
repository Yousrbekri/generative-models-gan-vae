# Generative Models Suite — GANs & VAE

Implémentation progressive des principales architectures de modèles génératifs pour la synthèse d'images, du DCGAN au StyleGAN, avec une exploration comparative des VAE.

> Projet réalisé dans le cadre du module *Generative AI* — ING4 Data Science, USTO-MB.

## Contenu

| # | Modèle | Framework | Dataset | Points clés |
|---|--------|-----------|---------|--------------|
| 1 | **DCGAN** | TensorFlow/Keras | LEGO Bricks (64×64) | Convolutions, BatchNorm, entraînement adversarial |
| 1 | **WGAN / WGAN-GP** | TensorFlow/Keras | LEGO Bricks (64×64) | Distance de Wasserstein, weight clipping vs gradient penalty |
| 2 | **cGAN** | TensorFlow/Keras | MNIST (28×28) | Génération conditionnelle par label (one-hot) |
| 3 | **StyleGAN** (simplifié) | PyTorch | CIFAR-10 (64×64) | Mapping network, espace latent W, AdaIN, style mixing |
| 4 | **VAE** (2D & 20D) | PyTorch | MNIST (28×28) | ELBO, reparameterization trick, interpolation latente |



## Points techniques marquants

- **DCGAN** : générateur/discriminateur convolutifs, `tanh`/`sigmoid`, Adam(lr=2e-4, β1=0.5) pour limiter les oscillations adversariales.
- **WGAN** : remplacement de la BCE par la distance de Wasserstein-1, contrainte 1-Lipschitz via weight clipping (`c=0.01`), `n_critic=5`.
- **WGAN-GP** : gradient penalty (λ=10) sur des images interpolées, plus stable que le clipping, BatchNorm évitée dans le critic.
- **cGAN** : concaténation du label one-hot au bruit latent (générateur) et à l'image (discriminateur) pour un contrôle explicite de la classe générée.
- **StyleGAN** : mapping network (8 couches) transformant Z en un espace W plus disentangled, injection de style via AdaIN à chaque résolution, tenseur constant appris en entrée du générateur.
- **VAE** : perte ELBO = reconstruction (BCE) + KL divergence, reparameterization trick pour rendre l'échantillonnage différentiable, comparaison d'un espace latent 2D (visualisable directement) et 20D (visualisé via t-SNE).

## Synthèse GAN vs VAE

| Critère | GAN | VAE |
|---|---|---|
| Approche | Jeu adversarial | Maximisation de la vraisemblance (ELBO) |
| Qualité d'image | Nette | Plus floue |
| Espace latent | Non structuré | Structuré, continu, interpolable |
| Stabilité | Sensible (mode collapse) | Entraînement stable |

## Stack

`TensorFlow / Keras` · `PyTorch` · `NumPy` · `Matplotlib` · Kaggle GPU (P100/T4)

## Lancer les notebooks

```bash
pip install tensorflow torch torchvision numpy matplotlib
jupyter notebook notebooks/
```

## Rapport complet

Le rapport détaillé (architectures, hyperparamètres, courbes, analyses) est disponible dans [`report/rapport_tps_gai.pdf`](./report/rapport_tps_gai.pdf).
