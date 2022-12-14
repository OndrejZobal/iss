Tato bakalářská práce se zabývala vytvořením materiálu v prostředí Jupyter notebook do předmětu ISS. 
Předlohou těchto notebooků jsou materiály dostupné zde:

https://www.fit.vutbr.cz/study/courses/ISS/public/proj_studijni_etapa/  (MATLAB + PDF)
https://www.fit.vutbr.cz/~izmolikova/ISS/project/  (Jupyter notebooky)


Obrázky přidané do cvičení jsou stáhnuty z:
Unequalized_Hawkes_Bay_NZ.jpg   https://cs.m.wikipedia.org/wiki/Soubor:Unequalized_Hawkes_Bay_NZ.jpg
franceska.jpg   https://gwent.pro/card/1194&v=2453

WAV soubory přidané do cvičení stáhnuty z:
https://users.cs.cf.ac.uk/Dave.Marshall/CM0268/Lecture_Examples/


Nekteří studenti z testovací skupiny se dříve setkali s Jupyter notebooky v předmětu IZV. Notebooky jim byli poskytnuty odkazem na sdílený disk.
Studentům kteří se účastnili uživatelského testování byl uveden tento doporučený postup pro spuštění notebooků:

* Stáhněte si distribuci Pythonu Anaconda, https://www.anaconda.com/products/distribution, nainstalují se vám aplikace.
* Aktualizujte si verzi `matplotlib` knihovny na 3.5.1 a vyšší v programu Anaconda Navigator - v sekci Environtments, (stačí například zvolit base root)
* Dále je potřeba manuálně doinstalovat balíček `soundfile` pro cvičení 3 a 6. Spusťte si Anaconda Prompt, spustě příkaz: 'pip install SoundFile'

Responzivita UI widgetů se může v různých prostředí hodně lišit, ale není ve vytvořených příkladech poměrně důležitá. Doporučuji také vypnout všechna rozšíření pracující s UI, které máte v prohlížeči, pokud pracujete ve webovém rozhraní. Případně můžete zkusit IDE VSCode, pokud si stáhnete rozšíření pro podporu Jupyter souborů.
Pokud se vám ve vašem prostředí zdají generované výstupy například příliž velké, nastavte si parameter `figure.dpi` na rozumnou hodnotu.

from matplotlib import rcParams
rcParams['figure.dpi'] = 42  # vhodně zvolte

Funkce uvedene v noteboocích s prefixem `ipy_*` si nemusíte procházet, jsou často uvedeny přímo v noteboocích pro ukázku implementace vytvořeného widgetu a na konzultacích se často ukázalo, že je lepší mít přímo v notebooku pro hledání chyb (oblášt při použití webového prohlížeče) způsobené odlišným prostředím.
Neprocházet ani skripty `.py`, obsahují jen některé pomocné vykreslovací funkce.

Pokud si stihnete projít alespoň 80% notebooku, můžete uvést hodnocení v dotazníku.

