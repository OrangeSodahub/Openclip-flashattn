# Openclip-flashattn

Test model (on RTX3080): `ViT-L-14::laion2b-s32b-b82k` (possibly not correct)

--------------------------------------------------------------
shape   | baseline(ms)        | flash_attn(ms)  | speed up (x)
--------|---------------------|-----------------|-------------
(1, 77) | 24.572              | 21.433          | 1.146
(2, 77) | 26.026              | 22.943          | 1.134
(4, 77) | 26.306              | 23.107          | 1.138
(8, 77) | 26.326              | 23.323          | 1.128
(16, 77)| 38.338              | 36.686          | 1.045
