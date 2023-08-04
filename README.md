# HuBMAP Top50 Solution



Training: 1211 images from ds2 + 1000 from new crop ds1

Validation: 422 images from ds1

Image_size : 1280 

Pretrian model: yolov8l-seg.pt / yolov8x-seg.pt

![ffc68cbcab03d6f8d5999b8a1b3757f](https://github.com/WPR001/HuBMAP_solution/assets/77914093/306a35ed-a421-47da-b94c-52254b2312bc)

| Experiment      | Public LB | Private LB |
| --------------- | --------- | ---------- |
| Dilate          | 0.524     | 0.453      |
| Not with dilate | 0.496     | 0.5        |

