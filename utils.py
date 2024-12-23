import numpy as np
import math
import heapq
from collections import defaultdict, Counter

# RLE Compression
def rle_encode(data):
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append(data[i - 1] + str(count))
            count = 1
    encoded.append(data[-1] + str(count))  
    return ''.join(encoded)

def rle_decode(data):
    decoded = []
    i = 0
    while i < len(data):
        char = data[i]
        count = int(data[i + 1])
        decoded.append(char * count)
        i += 2
    return ''.join(decoded)

def rle_compression_ratio(data, encoded):
    original_size = len(data) * 8  
    max_count = max(int(encoded[i + 1]) for i in range(0, len(encoded), 2))
    bits_for_count = max_count.bit_length() 
    compressed_size = len(encoded) // 2 * (8 + bits_for_count)

    return original_size / compressed_size
# huffman compression
def huffman_encode(text):
    frequency = Counter(text)
    heap = [[weight, [char, ""]] for char, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_codes = sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))

    huffman_dict = {char: code for char, code in huffman_codes}
    print("Huffman Codes:", huffman_dict)


    encoded_text = "".join(huffman_dict[char] for char in text)
    return encoded_text, huffman_dict

def huffman_decode(encoded_text, huffman_dict):
    reverse_huffman_dict = {code: char for char, code in huffman_dict.items()}  
    decoded_text = ""
    temp = ""
    for bit in encoded_text:
        temp += bit
        if temp in reverse_huffman_dict:
            decoded_text += reverse_huffman_dict[temp]
            temp = ""  
    return decoded_text


# Golomb Compression
# def golomb_encode(data, m):
#     def unary_encode(n):
#         return '1' * n + '0'
    
#     def binary_encode(n, m):
#         return bin(n)[2:].zfill(m)
    
#     def golomb_code(n, m):
#         q = n // m
#         r = n % m
#         return unary_encode(q) + binary_encode(r, int(math.log2(m)))
    
#     encoding = ''
#     for num in data:
#         encoding += golomb_code(num, m)
#     return encoding

# def golomb_decode(data, m):
#     def unary_decode(data):
#         return len(data) - 1
    
#     def binary_decode(data, m):
#         return int(data[:m], 2), data[m:]
    
#     def golomb_decode(data, m):
#         q = unary_decode(data)
#         r, data = binary_decode(data[q + 1:], int(math.log2(m)))
#         return q * m + r, data
    
#     decoded = []
#     while data:
#         num, data = golomb_decode(data, m)
#         decoded.append(num)
#     return decoded
class GolombEncoding:
    def __init__(self, m):
        self.m = m

    def encode(self, n):
        q = n // self.m  # Quotient
        r = n % self.m   # Remainder

        # Unary code for the quotient
        unary_code = '1' * q + '0'

        # Calculate the number of bits for the binary representation of the remainder
        b = math.ceil(math.log2(self.m))
        cutoff = (1 << b) - self.m
        if r < cutoff:
            binary_code = format(r, f'0{b - 1}b')
        else:
            binary_code = format(r + cutoff, f'0{b}b')

        return unary_code + binary_code

    def decode(self, encoded_str):
        if not encoded_str:  # Check if string is empty
            return None, ""
            
        # Decode unary part to get the quotient
        q = 0
        while q < len(encoded_str) and encoded_str[q] == '1':
            q += 1
            
        # Check if we have enough characters left for the separator
        if q >= len(encoded_str):
            return None, encoded_str
            
        # Move past the '0' separator after unary code
        remaining_str = encoded_str[q + 1:]
        if not remaining_str:  # Check if we have anything after the separator
            return None, encoded_str
            
        # Calculate the number of bits for the binary part
        b = math.ceil(math.log2(self.m))
        cutoff = (1 << b) - self.m
        
        # Make sure we have enough bits left for the remainder
        if len(remaining_str) < b - 1:
            return None, encoded_str
            
        try:
            # Extract the binary part
            remainder_bits = b - 1 if len(remaining_str) >= b - 1 and int(remaining_str[:b - 1], 2) < cutoff else b
            if len(remaining_str) < remainder_bits:
                return None, encoded_str
                
            remainder_str = remaining_str[:remainder_bits]
            r = int(remainder_str, 2)
            
            if remainder_bits == b:
                r -= cutoff
                
            # Calculate and return the original integer
            return q * self.m + r, remaining_str[remainder_bits:]
            
        except ValueError:  # Handle invalid binary strings
            return None, encoded_str
def golomb_encode(data, m):
    encoder = GolombEncoding(m)
    return ''.join(encoder.encode(num) for num in data)

def golomb_decode(encoded_str, m):
    decoder = GolombEncoding(m)
    decoded = ''
    remaining_str = encoded_str
    while remaining_str:
        num, remaining_str = decoder.decode(remaining_str)
        if num is None:
            return None
        decoded += chr(num)
    return decoded

# Arithmatic Compression
class ArithmeticEncoding:
    def __init__(self, probabilities = {}):
        self.probabilities = dict()
        self.cumulative_probs = dict()

    def _calculate_cumulative_probabilities(self):
        cumulative = {}
        total = 0
        for symbol, prob in self.probabilities.items():
            cumulative[symbol] = (total, total + prob)
            total += prob
        return cumulative
    
    def _calculate_probabilities(self, message):
        frequency = {}
        total_characters = len(message)

        for char in message:
            frequency[char] = frequency.get(char, 0) + 1

        probabilities = {char: count / total_characters for char, count in frequency.items()}
        return probabilities

    def encode(self, message):
        self.probabilities = self._calculate_probabilities(message)
        self.cumulative_probs = self._calculate_cumulative_probabilities()
        
        low, high = 0.0, 1.0
        for symbol in message:
            range_width = high - low
            symbol_low, symbol_high = self.cumulative_probs[symbol]
            high = low + range_width * symbol_high
            low = low + range_width * symbol_low
        return (low + high) / 2, self.probabilities, len(message) # Return encoded_value, prob_dist, message_len

    def decode(self, encoded_value, message_length, probabilities = None):
        if probabilities:
            self.probabilities = probabilities
            self.cumulative_probs = self._calculate_cumulative_probabilities()
        
        decoded_message = []
        for _ in range(message_length):
            for symbol, (low, high) in self.cumulative_probs.items():
                if low <= encoded_value < high:
                    decoded_message.append(symbol)
                    range_width = high - low
                    encoded_value = (encoded_value - low) / range_width
                    break
        return ''.join(decoded_message)
    
def arithmetic_encode(message):
    encoder = ArithmeticEncoding()
    encoded_value, probabilities, message_length = encoder.encode(message)
    return encoded_value, probabilities, message_length

def arithmetic_decode(encoded_value, message_length, probabilities):
    decoder = ArithmeticEncoding()
    return decoder.decode(encoded_value, message_length, probabilities)

def compress_image(image, num_levels, full_scale=255):
    step_size = (full_scale + 1) / num_levels
 
    quantized_indices = np.floor(image / step_size).astype(int)
    compressed_image = (quantized_indices + 0.5) * step_size

    return compressed_image.clip(0, full_scale).astype(np.uint8), step_size, quantized_indices

def decompress_image(quantized_indices, step_size, num_levels):
    levels = np.arange(num_levels)
    reconstruction_levels = step_size * (levels + 0.5)
    quantized_indices = np.array(quantized_indices)
    decompressed_data = reconstruction_levels[quantized_indices]

    return decompressed_data.clip(0, 255).astype(np.uint8)



