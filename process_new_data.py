import os
import argparse
import numpy as np
from subprocess import run
from scipy.io import wavfile
from intervaltree import IntervalTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mido


python_call_ver = "python3"

class Note:
    start_time = 0
    end_time = 0
    instrument = 1
    note = 0
    start_beat = 0.0
    end_beat = 0.0
    note_value = "Unknown"
    
    def __str__(self):
        return str(int(self.start_time)) + ", " + str(int(self.end_time)) + ", " + str(self.instrument) + ", " + str(self.note) + ", " + str(self.start_beat) + ", " + str(self.end_beat) + ", " + self.note_value
    
    def __repr__(self):
        return str(int(self.start_time)) + ", " + str(int(self.end_time)) + ", " + str(self.instrument) + ", " + str(self.note) + ", " + str(self.start_beat) + ", " + str(self.end_beat) + ", " + self.note_value
    
    
def convert_maestro_dataset_midi(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file[len(file) - 4 :] == "midi":
                full_path = os.path.join(root, file)
                print(file)
                
                midi = mido.MidiFile(full_path, clip=True)
                
                #print("start")
                #with open(os.path.join(root, "midi_info.txt"), 'w') as f:
                #    f.write(str(midi))
                #print("end")
            
                notelist = []
                bpm = 0
                current_ticks = 0
                ticks_per_beat = midi.ticks_per_beat
                
                unknown_ctr = 0
                
                for msg in midi:
                    if msg.is_meta and msg.type == "set_tempo":
                        bpm = mido.tempo2bpm(msg.tempo)
                
                
                    current_ticks += msg.time * 2 * ticks_per_beat # no clue why, but msg.time is exactly the midi time value / 960. This recitifies it, still no clue why this happens
                    current_beats = float(current_ticks / ticks_per_beat)
                    
                    current_seconds = (current_beats / bpm) * 60
                    current_timestamp = current_seconds * 44100 # they want the sample # I guess. IDK why musicnet formatted like this.
                    
                    
                    if msg.type == "note_off":
                        print("off")
                        
                    
                    if msg.type == "note_on":
                        if msg.velocity > 0:
                            # note activated
                            newNote = Note()
                            newNote.note = msg.note
                            newNote.start_beat = current_beats
                            newNote.start_time = current_timestamp
                            notelist.append(newNote)
                            
                        elif msg.velocity == 0:
                            # note deactivated
                            
                            # find last recorded activated note with same note value and record end_beat for items
                            
                            for note in reversed(notelist):
                                if note.note == msg.note:
                                    if note.end_beat != 0.00:
                                        print("error - two consecutive off notes: " + str(note.note) + " " + str(note.end_beat)) # shouldn't be an issue even if it happens, but if it does something is wrong
                                        #f = open("temp_notelist.txt", "w")
                                        #for note_ in notelist:
                                        #    f.write(str(note_) + "\n")
                                        #f.close()
                                        
                                    else:
                                        note.end_beat = current_beats - note.start_beat
                                        note.end_time = current_timestamp
                                        
                                        if note.end_beat > 0.0208 and note.end_beat <= 0.0521:	    note.note_value ="Triplet Thirty Second"
                                        elif note.end_beat > 0.0521 and note.end_beat <= 0.0729:	note.note_value ="Sixty Fourth"
                                        elif note.end_beat > 0.0729 and note.end_beat <= 0.1042:	note.note_value ="Triplet Sixteenth"
                                        elif note.end_beat > 0.1042 and note.end_beat <= 0.1458:	note.note_value ="Thirty Second"
                                        elif note.end_beat > 0.1458 and note.end_beat <= 0.1771:	note.note_value ="Triplet Eighth"
                                        elif note.end_beat > 0.1771 and note.end_beat <= 0.2188:	note.note_value ="Dotted Thirty Second"
                                        elif note.end_beat > 0.2188 and note.end_beat <= 0.2917:	note.note_value ="Sixteenth"
                                        elif note.end_beat > 0.2917 and note.end_beat <= 0.3542:	note.note_value ="Triplet"
                                        elif note.end_beat > 0.3542 and note.end_beat <= 0.4375:	note.note_value ="Dotted Sixteenth"
                                        elif note.end_beat > 0.4375 and note.end_beat <= 0.5625:	note.note_value ="Eighth"
                                        elif note.end_beat > 0.5625 and note.end_beat <= 0.6875:	note.note_value ="Tied Eighth-Thirty Second"
                                        elif note.end_beat > 0.6875 and note.end_beat <= 0.875:	    note.note_value ="Dotted Eighth"
                                        elif note.end_beat > 0.875 and note.end_beat <= 1.0625:	    note.note_value ="Quarter"
                                        elif note.end_beat > 1.0625 and note.end_beat <= 1.1875:	note.note_value ="Tied Quarter-Thirty Second"
                                        elif note.end_beat > 1.1875 and note.end_beat <= 1.35:	    note.note_value ="Tied Quarter-Sixteenth"
                                        elif note.end_beat > 1.4 and note.end_beat <= 1.6:	        note.note_value ="Dotted Quarter"
                                        elif note.end_beat > 1.9 and note.end_beat <= 2.1:	        note.note_value ="Half"
                                        elif note.end_beat > 2.025 and note.end_beat <= 2.225:	note.note_value ="Tied Half-Thirty Second"
                                        elif note.end_beat > 2.15 and note.end_beat <= 2.35:	note.note_value ="Tied Half-Sixteenth"
                                        elif note.end_beat > 2.4 and note.end_beat <= 2.6:	note.note_value ="Tied Half-Eighth"
                                        elif note.end_beat > 2.9 and note.end_beat <= 3.1:	note.note_value ="Dotted Half"
                                        elif note.end_beat > 3.9 and note.end_beat <= 4.1:	note.note_value ="Whole"
                                        else:
                                            #print("unknown note duration " + str(note.end_beat) + " " + str(note.note))
                                            unknown_ctr += 1
                                            note.note_value = "Unknown"
                                        break

                                    
                                    
                        else:
                            print("negative velocity, error")
                
                if bpm == 0:
                    print("error - set_tempo message not found")
                
                
                #print(midi)
                #with open(os.path.join(root, "midi_info.txt"), 'w') as f:
                #    f.write(str(midi))
                #break
                
                # write data to CSV
                
                f = open(os.path.join(root,file[:len(file) - 5] + ".csv"), "w")
                #print(os.path.join(root,file[:len(file) - 5] + ".csv"))
                f.write("start_time,end_time,instrument,note,start_beat,end_beat,note_value\n")
                for note in notelist:
                    f.write(str(note) + "\n")
                f.close()
                
                print(f"unknown notes: {unknown_ctr}, total notes: {len(notelist)}, %age: {100* unknown_ctr / len(notelist)}")
                #break
            
                
            
convert_maestro_dataset_midi("/path/to/maestro")