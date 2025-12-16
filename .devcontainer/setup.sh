sudo apt get update
sudo apt install -y libgl1 libglib2.0-0

# URL is invalid
# @see https://github.com/Deci-AI/super-gradients/pull/2061
sed -i 's/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/' /usr/local/python/3.9.25/lib/python3.9/site-packages/super_gradients/training/pretrained_models.py
sed -i 's/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/' /usr/local/python/3.9.25/lib/python3.9/site-packages/super_gradients/training/utils/checkpoint_utils.py

pip install -qq super-gradients==3.7.1
