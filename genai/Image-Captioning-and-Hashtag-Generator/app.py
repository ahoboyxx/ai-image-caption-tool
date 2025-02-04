import streamlit as st
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
from PIL import Image
import torch
import random

# Model initialization
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def creative_caption_generator(description):
    # Predefined caption templates with placeholders
    caption_templates = [
        "Capturing the essence of {description} 📸✨",
        "Just another day of {description} magic 🌟",
        "Living my best {description} moment 🌈",
        "Embracing the {description} vibes today 💫",
        "Snapshot of pure {description} joy 📷"
    ]
    
    # Randomly select and format captions
    return [template.format(description=description) for template in random.sample(caption_templates, 3)]

def hashtag_generator(description):
    # Generate creative hashtags based on description
    base_hashtags = [
        f"#{description.replace(' ', '')}Moment",
        f"#{description.replace(' ', '')}Life",
        f"#{description.replace(' ', '')}Vibes",
        f"#Exploring{description.replace(' ', '')}",
        f"#Love{description.replace(' ', '')}"
    ]
    
    # Add some generic, fun hashtags
    generic_hashtags = [
        "#LiveAndExplore", "#MomentsCaptured", "#Adventure", 
        "#Photography", "#InstaDaily"
    ]
    
    # Combine and randomize hashtags
    all_hashtags = base_hashtags + generic_hashtags
    return [" ".join(random.sample(all_hashtags, 10))]

def prediction(img_list):
    max_length = 30
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    img = []
    
    for image in tqdm(img_list):
        i_image = Image.open(image)
        st.image(i_image, width=200)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        img.append(i_image)

    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)

    output = model.generate(pixel_val, **gen_kwargs)
    predict = tokenizer.batch_decode(output, skip_special_tokens=True)
    return [pred.strip() for pred in predict]

def sample():
    sp_images = {
        'Sample 1': 'image/beach.png',
        'Sample 2': 'image/coffee.png', 
        'Sample 3': 'image/promotion.png'
    }
    
    cols = st.columns(3)
    
    for i, (name, sp) in enumerate(sp_images.items()):
        cols[i].image(sp, width=150)
        
        if cols[i].button("Generate", key=f"sample_{i}"):
            description = prediction([sp])
            if description:
                st.subheader("Description for the Image:")
                st.write(description[0])
                
                st.subheader("Captions for this image are:")
                captions = creative_caption_generator(description[0])
                for caption in captions:
                    st.write(caption)
                    
                st.subheader("#Hashtags")
                hashtags = hashtag_generator(description[0])
                for hash_tag in hashtags:
                    st.write(hash_tag)

def upload():
    with st.form("uploader"):
        image = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg","png","jpeg"])
        submit = st.form_submit_button("Generate")
        
        if submit and image:
            description = prediction(image)
            
            if description:
                st.subheader("Description for the Image:")
                for caption in description:
                    st.write(caption)
                    
                st.subheader("Captions for this image are:")
                captions = creative_caption_generator(description[0])
                for caption in captions:
                    st.write(caption)
                    
                st.subheader("#Hashtags")
                hashtags = hashtag_generator(description[0])
                for hash_tag in hashtags:
                    st.write(hash_tag)

def main():
    st.set_page_config(page_title="Caption and Hashtag Generation")
    st.title("Get Captions and Hashtags for your Image")
    
    tab1, tab2 = st.tabs(["Upload Image", "Sample"])
    
    with tab1:
        upload()

    with tab2:
        sample()
        
    st.subheader('By M.H.E.N')

if __name__ == '__main__': 
    main()