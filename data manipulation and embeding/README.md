<div id="top"></div>


<!--README -->
## README

[![Product Name Screen Shot][product-screenshot]](https://example.com)

I couldn't run your code for some reason, so I made my own data set from our data.
you can see it as excel files "data" and "forcast", that's how I thought the data is supposed to look like.

one thing needed to be done before slicing the data from the original excel is converting the WS and WD to X and Y elements.
this is done by using "point2xy" function with the following arguments:
 * dataset (one long pandas dataframe, before anything else is done on it) 
 * WS column name as a string
 * Wd column name as a string

after that is done we're supposed to reshape the data, that assumedly looks like the data in "data.xlsx"
next, the data and the forecast (that looks roughly the same, as seem in "forcast.xlsx") go into "Modifier" class (needed to be initialized with the data,forcast and total lan of desired output(in time))

you can find an example of the final output in "final.xlsx"
that's the data your'e supposed to work with  


<p align="right">(<a href="#top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->

[product-screenshot]: data_example.jpg
