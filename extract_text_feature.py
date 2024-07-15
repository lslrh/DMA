import os
from os.path import join, exists
import clip
from glob import glob

model_name = "ViT-L/14@336px"
print("Loading CLIP {} model...".format(model_name))
clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
print("Finish loading")

if isinstance(labelset, str):
    lines = labelset.split(',')
elif isinstance(labelset, list):
    lines = labelset
else:
    raise NotImplementedError

labels = []
for line in lines:
    label = line
    labels.append(label)
text = clip.tokenize(labels)
text = text.cuda()
text_features = clip_pretrained.encode_text(text)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
