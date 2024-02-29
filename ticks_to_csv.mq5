
#property copyright "Copyright 2000-2024, MetaQuotes Ltd."
#property version   "1.00"
#property script_show_inputs
#property strict
//--- Requesting 100 million ticks to be sure we receive the entire tick history
input long      getticks=100000000000; // The number of required ticks
string fileName = "ticks_data.csv";
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   int     attempts=0;     // Count of attempts
   bool    success=false;  // The flag of a successful copying of ticks
   MqlTick tick_array[];   // Tick receiving array
   MqlTick lasttick;       // To receive last tick data

   SymbolInfoTick(_Symbol,lasttick);
//--- Make 3 attempts to receive ticks
   while(attempts<3)
     {

      //--- Measuring start time before receiving the ticks
      uint start=GetTickCount();
      //--- Requesting the tick history since 1970.01.01 00:00.001 (parameter from=1 ms)
      long received=CopyTicks(_Symbol,tick_array,COPY_TICKS_ALL,1,getticks);

      // Check if ticks were successfully copied
      if(received > 0)
        {
         // Open the CSV file for writing
         int fileHandle = FileOpen(fileName, FILE_WRITE | FILE_CSV);

         // Check if the file was opened successfully
         if(fileHandle != INVALID_HANDLE)
           {
            // Write the CSV header
            FileWrite(fileHandle, "Time,Bid,Ask");

            // Write tick data to the CSV file
            for(long i = 0; i < received; i++)
              {
               string csvLine = StringFormat("%s,%.5f,%.5f", TimeToString(tick_array[i].time), tick_array[i].bid, tick_array[i].ask);
               FileWrite(fileHandle, csvLine);
              }


            // Close the CSV file
            FileClose(fileHandle);

            // Print success message
            Print("Downloaded ", received, " ticks for symbol ", _Symbol, " and period ", Period());
            Print("Ticks data saved to ", fileName);
           }
         else
           {
            // Print an error message if the file could not be opened
            Print("Failed to open the file for writing. Error code: ", GetLastError());
           }
        }
      else
        {
         // Print an error message if no ticks were downloaded
         Print("Failed to download ticks. Error code: ", GetLastError());
        }

      if(received!=-1)
        {
         //--- Showing information about the number of ticks and spent time
         PrintFormat("%s: received %d ticks in %d ms",_Symbol,received,GetTickCount()-start);
         //--- If the tick history is synchronized, the error code is equal to zero
         if(GetLastError()==0)
           {
            success=true;
            break;
           }
         else
            PrintFormat("%s: Ticks are not synchronized yet, %d ticks received for %d ms. Error=%d",
                        _Symbol,received,GetTickCount()-start,_LastError);
        }
      //--- Counting attempts
      attempts++;
      //--- A one-second pause to wait for the end of synchronization of the tick database
      Sleep(1000);
     }
  }