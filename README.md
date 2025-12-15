Main engine script is Chad_core_sim.py run it in the terminal with:

    cd ~/Desktop/Phys210Venv/manim-playground
source .venv/bin/activate

 ( replace file/path/with/yours )


Then you can plot with the diagnostic plotter or use ion_cloud_from_npz.py to
show animation ( Manim and Manim GL must be installed )]

  run with:

      manim -p -ql --renderer=opengl ion_cloud_from_npz.py ChadCore3D -o ChadCore3D_preview

IEC / POLYWELL

<img width="591" height="598" alt="Chad Core 3D" src="https://github.com/user-attachments/assets/fb3897f1-efd3-4b57-bf1b-3f803517dab4" />

POLYWELL

<img width="613" height="598" alt="Polywell 3D" src="https://github.com/user-attachments/assets/df30ffee-f153-4534-a88f-bfa71d8900d1" />
(Approximated by magnetic Quadrapole.)

  Render and save files with:

      manim --renderer=opengl --write_to_movie --disable_caching -qm \
      ion_cloud_from_npz.py ChadCore3D -o ChadCoreRun -p


      
IEC / POLYWELL

https://github.com/user-attachments/assets/a1bc403d-575f-4d89-9f0c-cbb70b58e4a8


IEC

https://github.com/user-attachments/assets/878b96df-0743-49d7-b6c1-9f91db35a9d6


POLYWELL

https://github.com/user-attachments/assets/a6827f3b-d4d3-492e-a1f4-5005b30ad244



The Chad Core is an idealized 3 dimensional version of the demo Fusor I constructed over the past summer

https://github.com/user-attachments/assets/2e0ae855-23b2-4068-942a-922a77dc68cf


Chad Core diagram

<img width="3485" height="2217" alt="image" src="https://github.com/user-attachments/assets/b3296f3d-0d21-4bb3-89a5-476e6f294de9" />

<img width="3263" height="3114" alt="image" src="https://github.com/user-attachments/assets/84b1a7b3-c8a9-4bc9-926b-3198a4f3ba92" />

