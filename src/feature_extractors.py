# # feature_extractors.py
# import torch
# import torch.nn as nn
# import timm  # biblioteca para modelos pré-treinados, incluindo ViT
# from torchvision import models

# # =========================
# # CNN FEATURE EXTRACTOR
# # =========================
# def extract_cnn_features(X_img, device=None):
#     """
#     Extrai features usando ResNet50 pré-treinada.
#     X_img: numpy array [N, 3, H, W]
#     Retorna: torch.Tensor [N, feature_dim]
#     """
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     # converter para tensor
#     X_tensor = torch.tensor(X_img, dtype=torch.float32).to(device)

#     # modelo pré-treinado
#     resnet = models.resnet50(pretrained=True)
#     # remover camada final (fc)
#     resnet = nn.Sequential(*list(resnet.children())[:-1])
#     resnet.eval().to(device)

#     with torch.no_grad():
#         feats = resnet(X_tensor)
#         feats = feats.view(feats.size(0), -1)  # flatten

#     return feats.cpu().numpy()


# # =========================
# # VIT FEATURE EXTRACTOR
# # =========================
# def extract_vit_features(X_img, device=None):
#     """
#     Extrai features usando Vision Transformer pré-treinado.
#     X_img: numpy array [N, 3, H, W]
#     Retorna: torch.Tensor [N, feature_dim]
#     """
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     # converter para tensor
#     X_tensor = torch.tensor(X_img, dtype=torch.float32).to(device)

#     # modelo pré-treinado ViT
#     vit = timm.create_model("vit_base_patch16_224", pretrained=True)
#     vit.reset_classifier(0)  # remove classificação
#     vit.eval().to(device)

#     with torch.no_grad():
#         feats = vit(X_tensor)
#         feats = feats.view(feats.size(0), -1)

#     return feats.cpu().numpy()
