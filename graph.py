import matplotlib.pyplot as plt
import mplfinance as mpf
from data_handling import pd,x,y,date

date = pd.to_datetime(date)

fig , (axis1 ,axis2) = plt.subplots(2,1, figsize=(12,10), sharex=True)

axis1.plot(date,x, color='blue',label='close')
plt.title("close data")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)

axis2.plot(date,y, color='red',label='open')
plt.xlabel('Date')
plt.ylabel('price')
plt.legend()
plt.title('Data viisualization')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()


