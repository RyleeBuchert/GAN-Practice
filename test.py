import sys
import time

# chars = 'ABCDEFGH'
# loop = range(1, len(chars)+1)

# LINE_CLEAR = '\x1b[2K'

# for idx in loop:
#     print(chars[:idx], end='\r')
#     time.sleep(0.5)

# print(end=LINE_CLEAR)
# print('done')


        
epochs = 5   
steps_per_epoch = 500

for epoch in range(epochs):
    for batch in range(steps_per_epoch):
        
        # Print progress
        if batch % (steps_per_epoch/20) == 0:
            string = f"Epoch {epoch+1}/{epochs}: ["
            for _ in range(int(batch//(steps_per_epoch/20)+1)):
                string += "="
            for _ in range(20 - int(batch//(steps_per_epoch/20)+1)):
                string += " "
            len_steps = len(str(steps_per_epoch))
            len_string = len(string) + 3 + (len_steps*2)
            string += f"] {batch}/{steps_per_epoch}"
            while len(string) != len_string:
                string += " "
            print(string, end='\r')
        batch += 1
        time.sleep(0.005)
