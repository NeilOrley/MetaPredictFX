//+------------------------------------------------------------------+
//|                                               Timer_modified.mq5 |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2021, Vladimir Karputov"
#property link      "https://www.mql5.com/ru/market/product/43516"
#property version   "1.000"

//--- display the window of input parameters when launching the script
#property script_show_inputs

//--- parameters for data reading
input string InpFileName="python_file.txt"; // file name
input string InpDirectoryName="Files"; // directory name
//---
#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>
//#include <File.mqh>
//---
CPositionInfo  m_position;                   // object of CPositionInfo class
CTrade         m_trade;                      // object of CTrade class
//--- input parameters
input uchar    InpAfterHour   = 0; // After: Hour ... (max 255)
input uchar    InpAfterMinutes= 42; // After: Minutes ... (max 255)
input uchar    InpAfterSeconds= 59; // After: Seconds ... (max 255)
//---
int file_handles;
uchar value_uchar;
long     m_after  = 0;
bool         started=false;     // flag of counter relevance
string str1,str2,str3,str4;



//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

int OnInit()
  {
  str1="C:/XXX/MetaQuotes/Terminal/XXXX/MQL5/Files/python_";
  str3="_file.txt";
  str2=Symbol();
  str4=str1+str2+str3;

   // Open file to read
   string file_path = str4;
   file_handles = FileOpen(file_path, FILE_READ|FILE_TXT);
   
   if(file_handles != INVALID_HANDLE)
     {
      PrintFormat("File %s opened correctly", file_path);
      
      // Read uchar value of the file
      if (FileReadInteger(file_handles, INT_VALUE))
        {
         PrintFormat("Uchar value read from file: %u", value_uchar);
         m_after  = 0;
         m_after=value_uchar;
        }
      else
        {
         Print("Error when reading the uchar value from file");
        }

      // Close file after read
      FileClose(file_handles);
     }
   else
     {
      m_after  = 0;
      m_after=InpAfterHour*60*60+InpAfterMinutes*60+InpAfterSeconds;
      PrintFormat("Error when opening file %s, error code = %d", file_path, GetLastError());
     }
     
     

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
//+------------------------------------------------------------------+
//| called when a Trade event arrives                                |
//+------------------------------------------------------------------+
void OnTrade()
  {
   if(started) SimpleTradeProcessor();
   
  }
  
 void SimpleTradeProcessor()
  {
  str1="C:/Users/XXX/MetaQuotes/Terminal/XXX/MQL5/Files/python_";
  str3="_file.txt";
  str2=Symbol();
  str4=str1+str2+str3;
     // Open file to read it
   string file_path = str4;
   file_handles = FileOpen(file_path, FILE_READ|FILE_TXT);
   
   if(file_handles != INVALID_HANDLE)
     {
      PrintFormat("File %s is opened correctly", file_path);
      
      // Reads the uchar value from the file
      if (FileReadInteger(file_handles, INT_VALUE))
        {
         PrintFormat("value_uchar read form file: %u", value_uchar);
         m_after  = 0;
         m_after=value_uchar;
        }
      else
        {
         Print("Erro while reading the value uchar from file");
        }

      // Closes the file after reading
      FileClose(file_handles);
     }
   else
     {
      m_after  = 0;
      m_after=InpAfterHour*60*60+InpAfterMinutes*60+InpAfterSeconds;
      PrintFormat("Error when opening the fileo %s, Code error = %d", file_path, GetLastError());
     }
     
  }