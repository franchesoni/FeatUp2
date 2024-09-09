# to download the dataset
cd /export/home/data/featupdata/
wget https://marhamilresearch4.blob.core.windows.net/stego-public/pytorch_data/cocostuff.zip


# results
trained for 100 epochs the linear layer and the prototypes

this is without feature upsampling, but with prediction upsampling instead:
{'Prototype Accuracy': 0.9843603372573853, 'Prototype mIoU': 0.3714086413383484, 'Linear Accuracy': 0.9864199161529541, 'Linear mIoU': 0.41732436418533325}

this is with bilinear feature upsampling:
{'Prototype Accuracy': 0.9848774075508118, 'Prototype mIoU': 0.381422758102417, 'Linear Accuracy': 0.9869206547737122, 'Linear mIoU': 0.4267471432685852}

if using featup:
{'Prototype Accuracy': 0.02263581193983555, 'Prototype mIoU': 0.33935728669166565, 'Linear Accuracy': 0.9870721697807312, 'Linear mIoU': 0.42194628715515137}