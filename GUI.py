from faulthandler import disable
from logging import raiseExceptions
import os
from tabnanny import filename_only
import tkinter as tk
from tkinter import *
import tkinter.filedialog as tk_f
import tkinter.scrolledtext as st
from typing import final
from PIL import Image, ImageTk
import sys
from utils import DataFactory, custom_collate
from models import SpatiallyConditionedGraph as SCG
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
import torch
import numpy as np

from utils import DataFactory

gui = tk.Tk()
gui.title('HOI Detection')
gui.geometry('1280x900')
gui.resizable(False, False)

class myGUI():
    def __init__(self) -> None:
        self.test_image_path = "hicodet/hifo_data/test"
        self.data_root = 'hicodet'
        self.detection_root = 'hicodet/detections/hifo_data_preprocessing/base_nms-0.4_tl-2_best-base-update/test'
        self.model_path = 'checkpoints/hifo/hifo_hoi_hico-finetunede_epoch-16_nms-0.4_ohem-loss_update.pt'
        #original filename location
        self.filename = ''
        #temp_result_filename: "HICO_test_xxxxxx"
        self.temp_result_filename = ''
        #temp_result_filename index
        self.filename_idx = 0
        #result filename location
        self.result_filename = ''
        #final HOI output
        self.final_output = ''

    def choose_file(self):
        self.filename = tk_f.askopenfilename(title='open an image file', multiple=False, initialdir = self.test_image_path)
        self.path.set(self.filename)
        img_open = Image.open(self.filename_entry.get())
        img = ImageTk.PhotoImage(img_open)
        #print(self.filename_entry.get())
        self.image_label_left.config(image=img)
        self.image_label_left.image = img
        #when retrieve new image, reset the previous output
        self.text_result.delete('1.0',END)
        self.final_output = ''
        
    def display_result(self):     
        #get the current opened filename
        self.temp_result_filename = self.filename[-20:]
        self.main()
        #retrieve saved image from file
        result_dir = "D:\Academic\FYP\\backup_20220308\spatially-conditioned-graphs\\"
        #print(result_filename)
        self.result_filename = result_dir + self.temp_result_filename
        img_open = Image.open(self.result_filename)
        img = ImageTk.PhotoImage(img_open)
        self.image_label_right.config(image=img)
        self.image_label_right.image = img
        
        self.text_result.insert(END, self.final_output)
        
        
    def set_window(self):
        self.path = StringVar()
        self.button_choose_file = tk.Button(gui, text = 'open file' ,command=self.choose_file)
        self.button_choose_file.place(x = 200, y = 700,w = 150, h = 30)
        self.filename_entry = Entry(gui, state='readonly', text=self.path,width=50)
        self.filename_entry.pack()
        
        self.image_label_left = Label(gui,bg = 'gray',text='open an original file')
        self.image_label_left.place(x=10, y=50, width = 640, height=640)  
        
        self.button_HOI = tk.Button(gui, text = 'HOI Detection',command=self.display_result)
        self.button_HOI.place(x = 960, y = 700,w = 150, h = 30)
        self.image_label_right = Label(gui, bg = 'white', text = 'display result')
        self.image_label_right.place(x=630, y=50, width = 640, height=640) 
        
        self.text_result = st.ScrolledText(gui, width = 165, height = 5) 
        self.text_result.pack()
        self.text_result.place(x=50,y=740)
        

    def colour_pool(self,n):
        pool = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#17becf', '#e377c2'
        ]
        nc = len(pool)
    
        repeat = n // nc
        big_pool = []
        for _ in range(repeat):
            big_pool += pool
        return big_pool + pool[:n%nc]


    def draw_boxes(self,ax,boxes,obj_class,labels,idxh,idxo):

        idxo = torch.unique(idxo, sorted=False)
        idxh = torch.unique(idxh, sorted=False)
        unique_h = len(idxh.tolist())
        sorted_boxes = torch.zeros_like(boxes)
        #count number of unique human detected
        
        count_box = 0
        human = 0
        
        unique_obj_classes = torch.unique(obj_class)
        obj_class = obj_class.tolist()
        unique_obj_classes = unique_obj_classes.tolist()
        
        if (unique_h<=1):
        #first, put human in the front
            for i in idxh:
                sorted_boxes[count_box] = boxes[i]
                count_box =+ 1
            #followed by all object
            for i in idxo:
                sorted_boxes[count_box]=boxes[i]
                count_box += 1
            #insert one human class '1000' at the front of the unique object list
            h_idx = 1000
            unique_obj_classes.insert(0,h_idx)
            #insert all human detections (class '1000') at the front of the object list
            human = 0
            while human < unique_h:
                obj_class.insert(0,h_idx)
                human += 1
                
        else:
            for i in idxo:
                sorted_boxes[count_box]=boxes[i]
                count_box += 1
            
        xy = sorted_boxes[:, :2].unbind(0)
        
        
        labels = labels.tolist()
        idxo=idxo.tolist()
        idxh=idxh.tolist()
    
        
        colour = self.colour_pool(len(unique_obj_classes))
        
        h, w = (sorted_boxes[:, 2:] - sorted_boxes[:, :2]).unbind(1)
        idx = 0
        while idx < len(unique_obj_classes):
            if (unique_h<=1):
                for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
                    if obj_class[i] == unique_obj_classes[idx]:
                        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor=colour[idx])
                        ax.add_patch(patch)
                        if i >= human:
                        #draw object bboxes
                            if labels[i-human] == 1:
                                txt_label_true = plt.text(*a.tolist(), str(idxo[i-1]), fontsize=10, fontweight='bold', color='w')
                                txt_label_true.set_path_effects([peff.withStroke(linewidth=2, foreground='#000000')])
                            else:
                                txt = plt.text(*a.tolist(), str(idxo[i-1]), fontsize=7, fontweight='semibold', color=colour[idx])  
                                txt.set_path_effects([peff.withStroke(linewidth=2, foreground='#ffffff')])
                        else:
                        #draw human bboxes
                            for human_label in idxh:
                                txt = plt.text(*a.tolist(), str(human_label), fontsize=7, fontweight='semibold', color=colour[idx])
                        plt.draw()
                idx += 1
            else:
                for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
                    if obj_class[i] == unique_obj_classes[idx]:
                        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor=colour[idx])
                        ax.add_patch(patch)
                        txt = plt.text(*a.tolist(), str(idxo[i]), fontsize=15, fontweight='semibold', color=colour[idx])
                        txt.set_path_effects([peff.withStroke(linewidth=2, foreground='#ffffff')])
                        plt.draw()
                idx += 1
            
    def visualise_entire_image(self,dataset, output):
        
        """Visualise bounding box pairs in the whole image by classes"""
        bh=output['boxes_h']
        bo=output['boxes_o']
        no = len(bo)
       
        bbox, inverse = torch.unique(torch.cat([bo, bh]), dim=0, return_inverse=True)
        idxh = inverse[no:]
        idxo = inverse[:no]
    
        im = dataset.dataset.load_image(
            os.path.join(
                dataset.dataset._root,
                dataset.dataset.filename(self.filename_idx)
            )
        )
    
        # Print predicted classes and scores
        scores = output['scores']
        prior = output['prior']
        index = output['index']
        pred = output['prediction']
        labels = output['labels']
        obj_class=output['object']
        unary_labels=output['unary_labels']
    
        unique_actions = torch.unique(pred)
        for verb in unique_actions:
            print(f"\n=> Action: {dataset.dataset.verbs[verb]}")
            sample_idx = torch.nonzero(pred == verb).squeeze(1)
            temp_verb = dataset.dataset.verbs[verb]
            temp_line = ''
            for idx in sample_idx:  
                b_idx = index[idx]
                print(
                    f"({idxh[b_idx].item():<2}, {idxo[b_idx].item():<2}),",
                    f"score: {scores[idx]:.4f}, prior: {prior[0, idx]:.2f}, {prior[1, idx]:.2f}",
                    f"label: {bool(labels[idx])}"
                )
                bh_idx = f"({idxh[b_idx].item():<2}, "
                bo_idx = f"{idxo[b_idx].item():<2}), "
                score = f"score: {scores[idx]:.4f}, "
                label = f"label: {bool(labels[idx])}, "
                if (labels[idx]):
                    print(dataset.dataset.verbs[verb] + ' ' + dataset.dataset.objects[obj_class[b_idx]])
                    print_label = dataset.dataset.verbs[verb] + ' ' + dataset.dataset.objects[obj_class[b_idx]]
                else:
                    print_label = ''
                temp_line = temp_line + temp_verb + bh_idx + bo_idx + score + label + print_label + "\n"
                    
            self.final_output = self.final_output + temp_line
            #print('final_output:')
            #print(self.final_output)
            
        # Draw the bounding boxes
        fig = plt.figure()
        plt.imshow(im)
        ax = plt.gca()

        if len(unary_labels) == len(labels):
            self.draw_boxes(ax, bbox,obj_class,labels,idxh,idxo)
        else:
            self.draw_boxes(ax, bbox,obj_class,unary_labels,idxh,idxo)
        save_name =  dataset.dataset.filename(self.filename_idx)
        plt.savefig(save_name)



    @torch.no_grad()
    def main(self):
    
        dataset = DataFactory(
            name='hifo', partition='test',
            data_root=self.data_root,
            detection_root=self.detection_root,
        )
        
        i=0
        while i < len(dataset):
            if self.temp_result_filename == dataset.dataset.filename(i):
                self.filename_idx = i
                break
            i = i + 1
    
        net = SCG(
            dataset.dataset.object_to_verb, 49, num_classes=120,
            num_iterations=2,
            box_score_thresh=0.2,
            box_nms_thresh=0.4,
            max_object=15
        )
        net.eval()
    
        checkpoint = torch.load(self.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
        
        
        image, detection, target = dataset[self.filename_idx]
        image = [image]; detection = [detection]; target = [target]
    
        output = net(image, detection, target)
        self.visualise_entire_image(dataset, output[0])

           
mygui = myGUI()
mygui.set_window()
gui.mainloop()



