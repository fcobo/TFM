using System;
using System.IO;
using System.Text;

using iTextSharp;
using iTextSharp.text;
using iTextSharp.text.pdf;
using iTextSharp.text.pdf.parser;
using iTextSharp.text.exceptions;



namespace wavespace
{
    
    public class Pdf
    {

        public Pdf(){

        }
    
        private PdfReader reader_;
        private PdfReaderContentParser parser_;

        public void setEncoding(){
            System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
        }


        public bool openPDF(string fileName){

            try{
                reader_ = new PdfReader(fileName);
                parser_ = new PdfReaderContentParser(reader_);
                return true;
            }
            catch{
                return false;
            }
        }

        public bool closePDF(){
            try{
                reader_.Close();
                return true;
            }
            catch{
                return false;
            }
        }

        public int getNumberPages(){
            return reader_.NumberOfPages;
        }

        public float getSizeTop(int page){
            Rectangle mediabox = reader_.GetPageSize(page);
            return mediabox.Top;
        }

        public float getSizeRight(int page){
            Rectangle mediabox = reader_.GetPageSize(page);
            return mediabox.Right;
        }


        public string readPDFbyArea(int page, float x, float y, float width, float height){

            StringBuilder text = new StringBuilder();
            //Rectangle rect = new Rectangle(0, 0, 0, 0);
            Rectangle rect = reader_.GetPageSize(page);

            rect.Bottom = rect.Top - y - height;//842 - 49; 841.92
            rect.Left = x;//168;;
            rect.Right = x + width;//168+198;;
            rect.Top = rect.Top - y;//842 - 31; 841.92

            RenderFilter[] filter = {new RegionTextRenderFilter(rect)};
            ITextExtractionStrategy strategy = new FilteredTextRenderListener(new LocationTextExtractionStrategy(), filter);

            string currentText = PdfTextExtractor.GetTextFromPage(reader_, page, strategy);
            if (!String.IsNullOrWhiteSpace(currentText))
            {
                currentText = Encoding.UTF8.GetString(ASCIIEncoding.Convert(Encoding.Default, Encoding.UTF8, Encoding.Default.GetBytes(currentText)));
                text.Append(currentText);
            }

            return text.ToString();
        }

        public string readPDFbyPage(int page){

            StringBuilder text = new StringBuilder();
            ITextExtractionStrategy strategy = new SimpleTextExtractionStrategy();

            string currentText = PdfTextExtractor.GetTextFromPage(reader_, page, strategy);

            if (!String.IsNullOrWhiteSpace(currentText)){
                            currentText = Encoding.UTF8.GetString(ASCIIEncoding.Convert(Encoding.Default, Encoding.UTF8, Encoding.Default.GetBytes(currentText)));
                            text.Append(currentText);
                            return text.ToString();
            }
            else{
                return "";
            }       
        }
    }
}
