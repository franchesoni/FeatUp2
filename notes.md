# to download the dataset
cd /export/home/data/featupdata/
wget https://marhamilresearch4.blob.core.windows.net/stego-public/pytorch_data/cocostuff.zip


# results
trained for 100 epochs the linear layer and the prototypes

this is without feature upsampling, but with prediction upsampling instead:
{'Prototype Accuracy': 0.9968215823173523, 'Prototype mIoU': 0.3235992193222046, 'Linear Accuracy': 0.9957858920097351, 'Linear mIoU': 0.20109277963638306}

this is with bilinear feature upsampling:
{'Prototype Accuracy': 0.9969209432601929, 'Prototype mIoU': 0.3321753144264221, 'Linear Accuracy': 0.9959864616394043, 'Linear mIoU': 0.21298973262310028}
