import streamlit as st
from io import BytesIO
from utils import *
import numpy as np
from PIL import Image



# Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["File Compression", "Image Compression"])
if page == "File Compression":
    st.title("File Compression and Decompression")

    uploaded_file = st.file_uploader("Choose a file", type=["txt"])
    method = st.selectbox("Select Compression Method", ["RLE", "Golomb", "Arithmetic", "Huffman"])
    if method == "Golomb":
        m = st.number_input("Enter value for m", min_value=1, value=10, key="m")

    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        text = st.text_area("File Content", file_content, height=200)
        compressed_data = None

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Compress"):
                if method == "RLE":
                    compressed_data = rle_encode(text)
                    cr = rle_compression_ratio(text, compressed_data)
                    st.write(f"Compression Ratio: {cr:.2f}")
                elif method == "Golomb":
                    compressed_data = golomb_encode([ord(char) for char in text], m)
                elif method == "Arithmetic":
                    encoded_value, probabilities, mess_len = arithmetic_encode(text)
                    compressed_data = str(encoded_value) + '\n' + str(probabilities) + '\n' + str(mess_len)
                elif method == "Huffman":
                    encoded, huffman_dict = huffman_encode(text)
                    compressed_data = encoded + '\n' + str(huffman_dict)

                st.text_area("Compressed Data", compressed_data, height=200)
                
                # Add download button for compressed data
                compressed_bytes = compressed_data.encode('utf-8')
                st.download_button(
                    label="Download Compressed File",
                    data=compressed_bytes,
                    file_name="compressed.txt",
                    mime="text/plain"
                )
                if method == "RLE":
                    decompressed_data = rle_decode(compressed_data)
                elif method == "Golomb":
                    decompressed_data = golomb_decode(compressed_data, m)
                elif method == "Arithmetic":
                    decompressed_data = arithmetic_decode(encoded_value, mess_len, probabilities)
                elif method == "Huffman":
                    decompressed_data = huffman_decode(encoded, huffman_dict)
                st.text_area("Decompressed Data", decompressed_data, height=200)

        with col2:
            if st.button("Decompress"):
                try:
                    if method == "RLE":
                        decompressed_data = rle_decode(text)
                    elif method == "Golomb":
                        decompressed_data = golomb_decode(text, m)
                    elif method == "Arithmetic":
                        lines = text.split('\n')
                        encoded_value = float(lines[0])
                        probabilities = eval(lines[1])
                        mess_len = int(lines[2])
                        decompressed_data = arithmetic_decode(encoded_value, mess_len,probabilities)
                    elif method == "Huffman":
                        encoded, huffman_dict = text.split('\n')
                        decompressed_data = huffman_decode(encoded, eval(huffman_dict))
                    st.text_area("Decompressed Data", decompressed_data, height=200)
                    decompressed_bytes = decompressed_data.encode('utf-8')
                    st.download_button(
                        label="Download Decompressed File",
                        data=decompressed_bytes,
                        file_name="decompressed.txt",
                        mime="text/plain"
                    )
                except:
                    st.error("Invalid file. Please upload a valid compressed file.")
                    st.stop()
                

elif page == "Image Compression":
    st.title("Image Compression and Decompression")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    method = st.selectbox("Select Compression Method", ["Uniform Quantization"])

    if uploaded_image is not None:
        original_image = uploaded_image.read()
        image = Image.open(BytesIO(original_image))
        image_array = np.array(image)
        if method == "Uniform Quantization":
            num_levels = st.number_input("Enter number of levels", min_value=1, value=4, max_value=256, key="num_levels")
        st.image(image, caption="Uploaded Image")
        type = st.selectbox("", ["compress", "decompress"])

        if type == "compress":
            quantized_indices = None
            if st.button("Compress"):
                st.write("Compressing...")
                compressed_image, step_size, quantized_indices = compress_image(image_array, num_levels)
                st.image(compressed_image, caption="Compressed Image")
                compressed_image = Image.fromarray(compressed_image)
                buffer = BytesIO()
                compressed_image.save(buffer, format=image.format)
                buffer.seek(0)
                st.download_button(
                    label="Download Compressed Image",
                    data=buffer,
                    file_name="compressed_image." + image.format.lower(),
                    mime="image/" + method.lower().split()[0]
                )
                inds = '\n'.join(map(str, quantized_indices.flatten())) 
                st.write('step size:', step_size)
                st.download_button(
                    label="Download the Quantized Indices",
                    data=inds,
                    file_name="quantized_indices.txt",
                    mime="text/plain"
                )
                st.write("Decompressed Image")
                decompressed_image = decompress_image(quantized_indices, step_size, num_levels)
                st.image(decompressed_image, caption="Decompressed Image")

        else:
            quantized_indices = st.file_uploader("Enter quantized indices", type=["txt"])
            if quantized_indices is not None:
                    try:
                        content = quantized_indices.read().decode("utf-8")  # Decode bytes to string
                        quantized_indices = np.array([int(x) for x in content.splitlines()])  # Convert each line to int
                        quantized_indices = quantized_indices.reshape(image_array.shape)
                    except:
                        st.error("Invalid file format. Please upload a valid txt file.")
                        st.stop()
                    step_size = st.number_input("Enter step size", min_value=1, value=1, key="step_size")
                    if st.button("Decompress"):
                        st.write("Decompressing...")
                        decompressed_image = decompress_image(quantized_indices, step_size, num_levels)
                        st.image(decompressed_image, caption="Decompressed Image")
                        decompressed_image = Image.fromarray(decompressed_image)
                        buffer = BytesIO()
                        decompressed_image.save(buffer, format=image.format)
                        buffer.seek(0)
                        
                        # Add download button for decompressed image
                        st.download_button(
                            label="Download Decompressed Image",
                            data=buffer,
                            file_name="decompressed_image." + method.lower(),
                            mime="image/" + method.lower()
                        )
