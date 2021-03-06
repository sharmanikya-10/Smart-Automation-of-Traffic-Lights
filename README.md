# Inspiration
Traffic-jam is a very big problem in developing cities, In fact it’s ever increasing day-by-day nature makes it difficult to find where the traffic density is more in real time, so that to schedule a better traffic signal control and effective traffic routing. The root cause of this can be of different situations like congestion in traffic like insufficient Road width, Road conditions due to weather, unrestrained demand, large delay of Red Light etc. While insufficient capacity and unrestrained demand are somewhere interrelated, the delay of respective light is hard coded and not dependent on traffic. Indeed, manual control is must, Therefore, in order to reduce man’s power, the need for simulating and optimizing traffic control to satisfy the increasing demand arises. Technology in  the recent past using image processing for surveillance and safety, which is widely used in vehicle and traffic management for traveler information. The traffic density estimation can also be achieved using Image Processing.

# Proposed Solution
We propose a technique that can be used for traffic control using image processing.  According to the traffic densities on all roads, our model will allocate smartly the time period of green light for each road. We have chosen image processing for calculation of traffic density as cameras are very much cheaper than other devices such as sensors. The proposed model is constructed as follows: We have a Raspberry Pi that is connected to 4 sets of LED that represent the traffic lights. It is the process of monitoring the traffic density of each side and  change the signal according to the density in every direction.

# Future Scope
This project can be enhanced in such a way as to control automatically the signals depending on the traffic density on the roads using sensors like IR detector/receiver module extended with automatic turn off when no vehicles are running on any side of the road which helps in power consumption saving. This project also allows better priority to certain vehicles like ambulances. when the IR detector receives this type of signal then it automatically transits to green. It acts as a life saving device.

# Demo
![image](https://user-images.githubusercontent.com/93609977/159646397-0fc6acd9-2105-4d28-8e70-ac75e4d7c351.png)
![image](https://user-images.githubusercontent.com/93609977/159650113-5cd09a42-444e-45e4-92e2-dfb55e1cb4cf.png)

# Prototype

![image](https://user-images.githubusercontent.com/93609977/159671021-e4a82baa-2303-4660-9503-349470e48288.png)
