//+------------------------------------------------------------------+
//|                                                        Timer.mq5 |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2021, Vladimir Karputov"
#property link      "https://www.mql5.com/ru/market/product/43516"
#property version   "1.000"
//---
#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>
//---
CPositionInfo  m_position;                   // object of CPositionInfo class
CTrade         m_trade;                      // object of CTrade class
//--- input parameters
input uchar    InpAfterHour   = 0; // After: Hour ... (max 255)
input uchar    InpAfterMinutes= 59; // After: Minutes ... (max 255)
input uchar    InpAfterSeconds= 59; // After: Seconds ... (max 255)
//---
long     m_after  = 0;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- forced initialization of variables
   m_after  = 0;
   m_after=InpAfterHour*60*60+InpAfterMinutes*60+InpAfterSeconds;
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   for(int i=PositionsTotal()-1; i>=0; i--) // returns the number of current positions
      if(m_position.SelectByIndex(i)) // selects the position by index for further access to its properties
        {
         if(TimeCurrent()-m_position.Time()>=m_after)
            m_trade.PositionClose(m_position.Ticket()); // close a position
        }
  }
//+------------------------------------------------------------------+