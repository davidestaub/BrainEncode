#!/usr/bin/env python3
# gpu_mini_feature_test_full_debug_fixed_v2.py
# -----------------------------------------------------------
# Works with: torch ≥ 2.1, transformers ≥ 4.40, matplotlib, ridge_utils
# -----------------------------------------------------------

import os, re
from collections import defaultdict

import numpy as np
import torch
torch.manual_seed(0); torch.cuda.manual_seed_all(0)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from pathlib import Path
import sys
#local_hf = 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


from ridge_utils.DataSequence import DataSequence

# ─────────────────────────────────────────────────────────────
# 0)  GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_NAME          = "meta-llama/Llama-3.3-70B-Instruct"
AVG_TR,  WPM        = 1.18, 150.0
LOOKBACK1, LOOKBACK2 = 4096, 8192
LAYER_IDX           = 31
MAX_SEQ_PER_BATCH   = 128
CTX_LIMIT_DEFAULT   = 2048

EVENT_MARKER        = "¶"
LP_SMOOTH_SIGMA     = 1.0      # Gaussian σ in *words*
Z_EPS               = 1e-6
DEBUG_TOPK          = True     # ← set False to silence top‑k read‑out
TOPK                = 5        # how many alternatives to print

LOOKAHEAD = [0]

bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

marot = """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte. Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser. Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen. ¶
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall. Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich. Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte. ¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp. Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“
„Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – „Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“¶
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops. Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging.
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte. Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“ ¶
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums. Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander. „Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste. ¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal. Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund: Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof. ¶
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen. Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.¶
Der Klub residierte in einer Jugendstilvilla am Zürichberg. Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren. Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin. Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden. ¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“ ¶
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer. ¶
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett. ¶
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen. Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben. ¶
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug."""

maupassant = """Guy de Maupassant Die Hand. Man drängte sich um den Untersuchungsrichter Bermutier, der seine Ansicht äußerte über den mysteriösen Fall in Saint Cloud. Seit einem Monat entsetzte dies unerklärliche Verbrechen Paris. Niemand konnte es erklären. Herr Bermutier stand, den Rücken gegen den Kamin gelehnt da, sprach, sichtete die Beweisstücke, kritisierte die verschiedenen Ansichten darüber, aber er selbst gab kein Urteil ab. Ein paar Damen waren aufgestanden, um näher zu sein, blieben vor ihm stehen, indem sie an den glattrasierten Lippen des Beamten hingen, denen so ernste Worte entströmten. Sie zitterten und schauerten ein wenig zusammen in neugieriger Angst und dem glühenden unersättlichen Wunsch nach Grauenhaftem, der ihre Seelen quälte und peinigte. Eine von ihnen, bleicher als die anderen, sagte während eines Augenblicks Stillschweigen : – Das ist ja schrecklich! Es ist wie etwas Übernatürliches dabei. Man wird die Wahrheit nie erfahren. Der Beamte wandte sich zu ihr : – Ja, gnädige Frau, wahrscheinlich wird man es nicht erfahren, aber wenn Sie von Übernatürlichem sprechen, so ist davon nicht die Rede. Wir stehen vor einem sehr geschickt ausgedachten und ungemein geschickt ausgeführten Verbrechen, das so mit dem Schleier des Rätselhaften umhüllt ist, daß wir die unbekannten Nebenumstände nicht zu entschleiern vermögen. Aber ich habe früher einmal selbst einen ähnlichen Fall zu bearbeiten gehabt, in den sich auch etwas Phantastisches zu mischen schien. Übrigens mußte man das Verfahren einstellen, da man der Sache nicht auf die Spur kam. Mehrere Damen sagten zu gleicher Zeit, so schnell, daß ihre Stimmen zusammenklangen : – Ach Gott, erzählen Sie uns das! Der Beamte lächelte ernst, wie ein Untersuchungsrichter lächeln muß, und sagte : – Glauben Sie ja nicht, daß ich auch nur einen Augenblick gemeint habe, bei der Sache wäre etwas Übernatürliches. Es geht meiner Ansicht nach alles mit rechten Dingen zu. Aber wenn sie statt ›übernatürlich‹ für das was wir nicht verstehen, einfach ›unaufklärbar‹ sagen, so wäre das viel besser. Jedenfalls interessierten mich bei dem Fall, den ich Ihnen erzählen werde, mehr die Nebenumstände. Es handelte sich etwa um folgendes : Ich war damals Untersuchungsrichter in Ajaccio, einer kleinen weißen Stadt an einem wundervollen Golf, der rings von hohen Bergen umstanden ist. Ich hatte dort hauptsächlich Vendetta - Fälle zu verfolgen. Es giebt wundervolle, so tragisch wie nur möglich, wild und leidenschaftlich. Dort kommen die schönsten Rächerakte vor, die man sich nur träumen kann, Jahrhunderte alter Haß, nur etwas verblaßt, aber nie erloschen. Unglaubliche Listen, Mordfälle, die zu wahren Massakren, sogar beinahe zu herrlichen Thaten ausarten. Seit zwei Jahren hörte ich nur immer von der Blutrache, diesem furchtbaren, korsischen Vorurteil, das die Menschen zwingt, Beleidigungen nicht bloß an der Person, die sie gethan, zu rächen, sondern auch an den Kindern und Verwandten. Ich hatte ihm Greise, Kinder, Vettern zum Opfer fallen sehen, ich steckte ganz voll solcher Geschichten. Da erfuhr ich eines Tages, daß ein Engländer auf mehrere Jahre eine im Hintergrund des Golfes gelegene Villa gemietet. Er hatte einen französischen Diener mitgebracht, den er in Marseille gemietet. Bald sprach alle Welt von diesem merkwürdigen Manne, der in dem Haus allein lebte und nur zu Jagd und Fischfang ausging. Er redete mit niemand, kam nie in die Stadt, und jeden Morgen übte er sich ein oder zwei Stunden im Pistolen - oder Karabiner - Schießen. Allerlei Legenden bildeten sich um den Mann. Es wurde behauptet, er wäre eine vornehme Persönlichkeit, die aus politischen Gründen aus seinem Vaterlande entflohen. Dann ging das Gerücht, daß er sich nach einem furchtbaren Verbrechen hier versteckt hielt ; man erzählte sogar grauenvolle Einzelheiten. Ich wollte in meiner Eigenschaft als Untersuchungsrichter etwas über den Mann erfahren, aber es war mir nicht möglich. Er ließ sich Sir John Rowell nennen. Ich begnügte mich also damit, ihn näher zu beobachten, und ich kann nur sagen, daß man mir nichts irgendwie Verdächtiges mitteilen konnte. Aber da die Gerüchte über ihn fortgingen, immer seltsamer wurden und sich immer mehr verbreiteten, so entschloß ich mich, einmal den Fremden selbst zu sehen, und ich begann regelmäßig in der Nähe seines Besitztums auf die Jagd zu gehen. Ich wartete lange auf eine Gelegenheit. Endlich bot sie sich mir dadurch, daß ich dem Engländer ein Rebhuhn vor der Nase wegschoß. Mein Hund brachte es mir, ich nahm es auf, entschuldigte mich Sir John Rowell gegenüber und bat ihn artig, die Beute anzunehmen. Er war ein großer, rothaariger Mann, mit rotem Bart, sehr breit und kräftig, eine Art ruhiger, höflicher Herkules. Er hatte nichts von der sprüchwörtlichen englischen Steifheit und dankte mir lebhaft für meine Aufmerksamkeit in einem englisch gefärbten Französisch. Nach vier Wochen hatten wir fünf oder sechs Mal zusammen gesprochen, und eines Abends, als ich an seiner Thür vorüberkam, sah ich ihn, wie er in seinem Garten rittlings auf einem Stuhl saß und die Pfeife rauchte. Ich grüßte, und er lud mich zu einem Glase Bier ein. Das ließ ich mir nicht zweimal sagen. Er empfing mich mit aller peinlichen englischen Artigkeit, sprach am höchsten Lobeston von Frankreich, von Korsika, und erklärte, er hätte dieses Eiland zu gern. Da stellte ich ihm mit größter Vorsicht, indem ich lebhaftes Interesse heuchelte, einige Fragen über sein Leben und über seine Absichten. Ohne Verlegenheit antwortete er mir, erzählte mir, er sei sehr viel gereist, in Afrika, Indien und Amerika und fügte lachend hinzu : – O, ich haben viele Abenteuer gehabt, o yes! Dann sprach ich weiter von der Jagd, und er erzählte mir interessante Einzelheiten über die Nilpferd -, Tiger -, Elephanten - und sogar Gorilla - Jagd. Ich sagte : – Alle diese Tiere sind gefährlich! Er lächelte : – O no, die schlimmste ist die Mensch! Er lachte gemütlich, in seiner behäbigen englischen Art und sagte : – Ich habe auch viel die Mensch gejagt! Dann sprach er von Waffen und forderte mich auf, bei ihm einzutreten, um ein paar Gewehre verschiedener Systeme zu besehen. Das Wohnzimmer war mit schwarzer, gestickter Seide ausgeschlagen, große, gelbe Blumen schlängelten sich über den dunklen Stoff und leuchteten wie Feuer. Er sagte : – Das ist japanische Stickerei! Aber mitten auf der größten Wand zog ein eigentümlicher Gegenstand meine Blicke auf sich. Von vier Ecken mit rotem Sammet umgeben, hob sich etwas Seltsames ab. Ich trat näher. Es war eine Hand. Eine menschliche Hand. Nicht die Hand eines Skelettes mit gebleichten, reinlich präparierten Knochen, sondern eine schwarze, vertrocknete Hand mit gelben Nägeln, bloßliegenden Muskeln und alten Blutspuren von dem glatt abgeschnittenen Knochen, als wäre er mitten im Unterarm mit einem Beile abgehackt. An dem Handgelenk war eine Riesen - Eisenkette befestigt, die mit einem so starken Ring, als wolle man einen Elephant daran binden, die Hand an der Mauer hielt. Ich fragte : – Was ist denn das? Der Engländer antwortete ganz ruhig : – Das war meine beste Feind ; sie kam von Amerika. Das ist mit die Säbel abgeschlagen und die Haut mit scharfe Kiesel abgekratzt und acht Tage in die Sonne getrocknet. Aho, sehr fein für mir! Ich faßte diese menschlichen Überreste, die einem Koloß angehört haben mußten, an. Diese Hand war gräßlich zu sehen, und unwillkürlich drängte sich mir der Gedanke an einen fürchterlichen Racheakt auf. Ich sagte : – Dieser Mann muß sehr stark gewesen sein! Der Engländer antworte ganz weich : – O yes, aber ich war stärker, ich hatte die Kette angebunden, sie zu halten. Ich meinte, er scherze und sagte : – Nun, diese Kette ist ja jetzt unnütz, die Hand wird ja nicht davon laufen. Sir John Rowell antwortete ernst : – Er wollte immer fortlaufen, die Kette war nötig. Mein Blick ruhte fragend auf seinem Gesicht, und ich sagte mir : Ist der Kerl verrückt, oder ist es ein schlechter Witz? Aber sein Gesicht blieb unbeweglich ruhig, voller Wohlwollen, er sprach von anderen Dingen, und ich bewunderte seine Gewehre. Aber ich bemerkte, daß geladene Revolver hier und da auf den Tischen lagen, als ob er in ständiger Furcht vor einem Angriff lebte. Ich besuchte ihn noch ein paar Mal, dann nicht mehr, man hatte sich an seine Anwesenheit gewöhnt, er war uns allen uninteressant geworden. Ein ganzes Jahr verstrich, da weckte mich eines Morgens, Ende September, mein Diener mit der Meldung, Sir John Rowell wäre in der Nacht ermordet worden. Eine halbe Stunde später betrat ich mit dem Gendarmerie - Hauptmann das Haus des Engländers. Der Diener stand ganz verzweifelt vor der Thür und weinte. Ich hatte zuerst den Mann in Verdacht, aber er war unschuldig. Den Schuldigen hat man nie entdecken können. Als ich in das Wohnzimmer des Sir John Rowell. trat, sah ich auf den ersten Blick mitten in dem Raum die Leiche auf dem Rücken liegen. Die Weste war zerrissen, ein Ärmel hing herab, alles deutete darauf hin, daß ein furchtbarer Kampf stattgefunden hatte. Der Engländer war erwürgt worden, sein schwarzes, gedunsenes Gesicht hatte etwas Gräßliches und schien ein furchtbares Entsetzen auszudrücken. Zwischen den zusammengebissenen Zähnen steckte etwas und sein blutiger Hals war von fünf Löchern durchbohrt, als wären fünf Eisenspitzen dort eingedrungen. Ein Arzt folgte uns, er betrachtete lange die Fingerspuren im Fleisch und that die seltsame Äußerung : – Das ist ja, als ob er von einem Skelett erwürgt worden wäre. Ein Schauder lief mir über den Rücken, und ich blickte zur Wand, auf die Stelle, wo ich sonst die entsetzliche Hand gesehen. Sie war nicht mehr da, die Kette hing zerbrochen herab. Da beugte ich mich zu dem Toten nieder und fand in seinem verzerrten Mund einen der Finger dieser verschwundenen Hand. Gerade am zweiten Glied von den Zähnen abgebissen, oder vielmehr abgesägt. Die Untersuchung wurde eingeleitet, man fand nichts, keine Thür war aufgebrochen worden, kein Fenster, kein Möbel. Die beiden Wachthunde waren nicht wach geworden. Die Aussage des Dieners war etwa folgende : Seit einem Monat schien sein Herr sehr erregt, er hatte viele Briefe bekommen, aber sie sofort wieder verbrannt. Oft nahm er in einem Wutanfall, fast tobsuchtartig, eine Reitpeische und schlug ein auf diese vertrocknete Hand, die an die Mauer geschmiedet und, man weiß nicht wie, zur Stunde, als das Verbrechen geschehen, geraubt worden war. Er ging sehr spät zu Bett und schloß sich jedesmal sorgfältig ein. Er hatte immer Waffen bei der Hand, manchmal sprach er Nachts laut, als zankte er sich mit jemandem. Diese Nacht hatte er aber zufällig keinen Lärm gemacht, und der Diener hatte Sir John erst ermordet vorgefunden, als er die Fenster öffnete. Er hatte niemandem im Verdacht. Was ich wußte, teilte ich dem Beamten und der Polizei mit, und auf der ganzen Insel wurde sorgfältig nachgeforscht – man entdeckte nichts. Da hatte ich eine Nacht, ein Vierteljahr nach dem Verbrechen, einen furchtbaren Traum. Es war mir, als sähe ich die Hand, die entsetzliche Hand wie einen Skorpion, wie eine Spinne längs der Vorhänge hinhuschen. Dreimal wachte ich auf, dreimal schlief ich wieder ein, dreimal sah ich dieses entsetzliche Überbleibsel um mein Zimmer herumjagen, indem es die Finger wie Pfoten bewegte. Am nächsten Tage brachte man mir die Hand, die man auf dem Kirchhof, wo Sir John Rowell begraben war, da man seine Familie nicht eruiert hatte, auf seinem Grabe gefunden hatte. Der Zeigefinger fehlte. Das, meine Damen, ist meine Geschichte, mehr weiß ich nicht. Die Damen waren bleich geworden, zitterten, und eine von ihnen rief : – Aber das ist doch keine Lösung und keine Erklärung, wir können ja garnicht schlafen, wenn Sie uns nicht sagen, was Ihrer Ansicht nach passiert ist. Der Beamte lächelte ernst : – O meine Damen, ich will Sie gewiß nicht um Ihre schönsten Träume bringen, ich denke ganz einfach, daß der Besitzer dieser Hand gar nicht tot war und daß er einfach gekommen ist, um sie mit der Hand wieder zu holen, die ihm übrig geblieben war ; aber ich weiß nicht, wie er das angestellt hat. Das wird eine Art Vendetta sein. Eine der Damen flüsterte : – Nein, das kann nicht so gewesen sein! Und der Untersuchungsrichter schloß immer noch lächelnd : – Ich habe es Ihnen doch gesagt, daß meine Erklärung Ihnen nicht passen würde."""

# ─────────────────────────────────────────────────────────────
# 1)  LITTLE HELPERS
# ─────────────────────────────────────────────────────────────
_WS_RE = re.compile(r"\s+")           # pre‑compiled for speed

#Situational models

from difflib import SequenceMatcher

def align_indices(ref_words, var_words, var_EB):
    """
    Translate EB indices that were recorded on a *variant* of the story
    (Finn, Maddox, …) so that they refer to the canonical `ref_words`.
    Works token‑exact: if spelling differs, the index is moved to the
    matching word in the reference text.
    """
    sm = SequenceMatcher(None, var_words, ref_words, autojunk=False)
    mapping = {}                       # var idx  →  ref idx
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            mapping.update({i1 + k: j1 + k for k in range(i2 - i1)})

    aligned = []
    for w in var_EB:
        ref_idx = mapping.get(w - 1)   # boundary is AFTER word w‑1
        if ref_idx is not None:
            aligned.append(ref_idx + 1)
    return aligned

def idx_before(boundary_list):
    """Return the word index *before* each boundary marker."""
    return [w - 1 for w in boundary_list]          # <<< NEW

def vram():
    free, tot = torch.cuda.mem_get_info()
    return f"{free/1e9:5.1f}/{tot/1e9:.1f} GB free"

def strip_eb_markers(txt: str):
    """
    1. Split on the literal EVENT_MARKER.
    2. Collapse *all* runs of whitespace to ONE space (same rule
       you apply to `raw_text` before feeding it to the model).
    3. Return
       - words : List[str]   (punctuation attached)
       - eb    : List[int]   (index AFTER which a marker occurred)
    """
    parts = txt.split(EVENT_MARKER)
    eb, words = [], []
    for i, seg in enumerate(parts):
        seg_norm = _WS_RE.sub(" ", seg.strip())
        seg_words = seg_norm.split(" ") if seg_norm else []
        words.extend(seg_words)
        if i < len(parts) - 1:          # not the last slice → a marker followed
            eb.append(len(words))
    return words, eb

def make_ds(words: list[str], avg_tr: float, wpm: float) -> DataSequence:
    dt    = 60.0 / wpm
    times = np.arange(len(words)) * dt
    tr    = np.arange(int(np.ceil(times[-1] / avg_tr))) * avg_tr
    return DataSequence(
        np.array(words),
        list(range(len(words))),
        times.astype(np.float32),
        tr.astype(np.float32),
    )

def last_or_none(seq):
    """safe max() – returns None for an empty iterable"""
    return max(seq) if seq else None

# ─────────────────────────────────────────────────────────────
# 2)  BOUNDARY DETECTION UTILS
# ─────────────────────────────────────────────────────────────
# Prompts


def is_single_token(s):
    return len(tok(s, add_special_tokens=False).input_ids) == 1

SYSTEM_PROMPT = (
    "Ein Event ist eine fortlaufende Situation. " 
    "Du bekommst gleich einen Text. "
    "Deine Aufgabe:\n"
    " Kopiere den Text wort‑für‑wort.\n"
    " Unterteile den Text in Events."
    " Füge ausschließlich den Event‑Marker ¶ ein – "
    " genau dann (und nur dann), wenn ein Event endet und ein neues beginnt.\n"
    " Halte die Zahl der Marker so klein wie möglich (<150 Marker pro 10000 Wörter).\n"
    "\n"
    "Wichtig\n"
    "• Gib nur den modifizierten Text zurück – keine Überschriften, Einleitungen, "
    "Erklärungen, Entschuldigungen oder sonstigen Kommentare.\n"
    "• Ändere keinerlei Rechtschreibung, Zeichensetzung oder Wortreihenfolge außer "
    "dem Einfügen von ¶.\n"
    "• Setze den Marker ohne zusätzliche Leer‑ oder Sonderzeichen (also nicht „**¶**“).\n"
)

#NOT USED ANYMORE
PROMPT_CORE = (
    "Ein Event ist eine fortlaufende Situation. "
    "Die folgende Geschichte muss kopiert und in möglichst wenige Events unterteilt werden."
    "Kopiere die folgende Geschichte wort für Wort und füge ausschlieslich den Event-Marker"
    f"{EVENT_MARKER} ein, dann und nur dann, wenn ein Event endet und ein neuer beginnt. "
    "\n\n"
    "Dies ist die Geschichte:\n\n"
)

##############################################################################
# Long‑story helper: handles stories that exceed the context window
##############################################################################
def greedy_copy_with_marker_smart(
    story: str,
    tok,
    model,
    *,
    device: str = "cuda",
    max_slack: int = 256,      # safety buffer for extra markers
    overlap: int = 50,         # tokens of look‑back on each new window
    event_marker: str = "¶",
):
    """
    Greedy copy of `story` with EVENT_MARKER inserted at event boundaries.
    Works for arbitrarily long stories by sliding a window over the text.

    Returns
    -------
    full_gen : str          # the story incl. inserted markers
    eb       : List[int]    # word indices (0‑based, whole story) AFTER which
                            # EVENT_MARKER was inserted
    """
    import math, re

    # ------------------------------------------------------------------ helpers
    def _generate_one_chunk(chunk_text: str):
        """Chat‑template wrapper for a single chunk (no length limit here)."""
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content":  chunk_text},
        ]
        ids_in = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        n_story_tok   = len(tok(chunk_text, add_special_tokens=False).input_ids)
        approx_markers = max(1, n_story_tok // 15)
        max_new       = n_story_tok + approx_markers + max_slack

        with torch.no_grad():
            out = model.generate(
                ids_in,
                do_sample=False,
                max_new_tokens=max_new,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][ids_in.size(1):], skip_special_tokens=False)

        # ----- extract marker offsets for this chunk -------------------------
        parts, eb_loc, w = re.split(f"({re.escape(event_marker)})", gen), [], 0
        for p in parts:
            if p == event_marker:
                eb_loc.append(w)          # boundary after word w‑1
            elif p.strip():
                w += len(p.split())
        return gen, eb_loc

    # ----------------------------------------------------------------- prework
    ctx_limit = getattr(model.config, "max_position_embeddings", 8192)

    # template length once (story inserted later, so use dummy)
    tmp = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": ""},
    ]
    tmpl_len = tok.apply_chat_template(
        tmp, add_generation_prompt=True, return_tensors="pt"
    ).size(1)

    room_for_story = ctx_limit - tmpl_len - max_slack
    if room_for_story <= 0:
        raise ValueError("Prompt itself already fills the context window.")

    # full story in tokens
    story_ids = tok(story, add_special_tokens=False).input_ids
    n_story_tokens = len(story_ids)

    # ---------------------------------------------------------------- dispatch
    if n_story_tokens <= room_for_story:             # short story → one pass
        return _generate_one_chunk(story)

    # ---------------------------------------------------------------- long case
    stride      = room_for_story - overlap
    n_chunks    = math.ceil((n_story_tokens - room_for_story) / stride) + 1

    words       = story.split()
    gen_chunks  = []
    eb_global   = []
    word_cursor = 0

    for ci in range(n_chunks):
        s_tok = ci * stride
        e_tok = min(s_tok + room_for_story, n_story_tokens)
        chunk_text = tok.decode(story_ids[s_tok:e_tok], skip_special_tokens=True)

        gen_chunk, eb_chunk = _generate_one_chunk(chunk_text)

        # keep markers outside trailing overlap (except last chunk)
        valid_limit = math.inf if ci == n_chunks - 1 else len(chunk_text.split()) - overlap
        eb_global.extend([word_cursor + w for w in eb_chunk if w < valid_limit])

        # merge generated text
        if ci < n_chunks - 1:
            keep = len(chunk_text.split()) - overlap
            gen_chunks.append(" ".join(gen_chunk.split()[:keep]))
            word_cursor += keep
        else:
            gen_chunks.append(gen_chunk)

    return " ".join(gen_chunks), eb_global


# ── 2) Revised boundary_lp with LOOK-AHEAD ───────────────────────────────

# ─────────────────────────────────────────────────────────────
# 2)  BOUNDARY‑LP WITH TRUE LOOK‑AHEAD   (replace the old function)
# ─────────────────────────────────────────────────────────────
def boundary_lp(ds, tok, model, lookahead: int, device="cuda"):
    """
    lp_word[i]  =  log p(EVENT_MARKER | context that includes *lookahead*
                 future tokens beyond word i)
    """
    enc          = tok(ds.data.tolist(),
                       is_split_into_words=True,
                       add_special_tokens=False,
                       return_tensors="pt")
    seq          = enc.input_ids[0].tolist()
    word_for_tok = enc.word_ids()

    lp_tok  = np.full(len(seq),     np.nan, np.float32)
    lp_word = np.full(len(ds.data), np.nan, np.float32)

    # ---- build the chat prefix once
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": PROMPT_CORE},
    ]
    chat_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt"
    ).tolist()[0]

    ctx_lim = getattr(model.config, "max_position_embeddings",
                      CTX_LIMIT_DEFAULT)
    room = ctx_lim - len(chat_ids) - 2                 # ‑2 safety margin
    if room <= lookahead:
        raise ValueError("look‑ahead larger than available context room")

    # ---- sliding window
    for s in range(0, len(seq), room - lookahead):
        e_main = min(s + room - lookahead, len(seq) - lookahead)
        block  = seq[s : e_main + lookahead]           # +future
        ids    = torch.tensor([chat_ids + block], device=device)

        with torch.no_grad():
            logits = model(ids,use_cache=False).logits[0]
        lps = torch.log_softmax(logits, -1).cpu().numpy()
        off = len(chat_ids)

        #
        # logits row j predicts token j+1
        # → to score boundary before token  j+lookahead,
        #   read logits at (j+lookahead‑1)
        #
        for j in range(0, len(block) - lookahead):  # j = position of w_t
            tgt_tok_idx = s + j  # absolute index of w_t
            #
            # prefix length that produced row = off + j
            # ⇒ row that predicts w_t is   off + (j - lookahead)
            #
            src_row = off + j #- lookahead
            score = lps[src_row, marker_id]
            lp_tok[tgt_tok_idx] = score
            w = word_for_tok[tgt_tok_idx]
            if w is not None:
                lp_word[w] = score

    mask = np.isnan(lp_word)
    if mask.all():
        raise RuntimeError("All scores are NaN — check look‑ahead logic.")
    if mask.any():
        idx_valid = np.where(~mask)[0]
        print("nan values at indices = ",np.where(mask)[0])
        # forward/backward fill the extremes
        first, last = idx_valid[0], idx_valid[-1]
        lp_word[:first] = lp_word[first]
        lp_word[last + 1:] = lp_word[last]
        # linear interpolation for interior gaps
        lp_word[mask] = np.interp(
            np.where(mask)[0], idx_valid, lp_word[idx_valid]
        )
        # lp_tok: simple nearest fill is enough for plotting
        mask_tok = np.isnan(lp_tok)
        if mask_tok.any():
            idx_valid = np.where(~mask_tok)[0]
            first, last = idx_valid[0], idx_valid[-1]
            lp_tok[:first] = lp_tok[first]
            lp_tok[last + 1:] = lp_tok[last]
            lp_tok[mask_tok] = np.interp(
                np.where(mask_tok)[0], idx_valid, lp_tok[idx_valid]
            )

    # ---- down‑sample word‑level to TR‑bins
    ds_lp = DataSequence(lp_word, ds.split_inds,
                         ds.data_times, ds.tr_times)
    lp_tr = np.asarray(
        ds_lp.chunksums("lanczos", window=3).data, dtype=np.float32
    )


    return lp_tr, lp_word, lp_tok


# ─────────────────────────────────────────────────────────────
# AUROC / PR METRICS, BOOTSTRAP & "PAPER" PLOTS (no sklearn)
# ─────────────────────────────────────────────────────────────
import math, json, datetime, random

# ---------- labels from raters (union, with tolerance) ----------
def build_labels_union(EB_by_rater, n_words, tol_words=0):
    pos = set()
    for EB_list in EB_by_rater.values():
        for w in EB_list:
            j = max(0, w-1)  # EB is after w-1 → score at word j
            pos.add(j)
    if tol_words > 0 and pos:
        dil = set()
        for j in pos:
            a, b = max(0, j - tol_words), min(n_words-1, j + tol_words)
            dil.update(range(a, b+1))
        pos = dil
    y = np.zeros(n_words, dtype=np.int8)
    if pos:
        y[list(pos)] = 1
    return y

# ---------- rank utils / AUROC ----------
def _rankdata_with_ties(x):
    # average ranks for ties, 1-based
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and x[order[j+1]] == x[order[i]]:
            j += 1
        r = (i + j + 2) / 2.0
        ranks[order[i:j+1]] = r
        i = j + 1
    return ranks

def auroc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n1 = int(labels.sum()); n0 = int((1-labels).sum())
    if n1 == 0 or n0 == 0: return np.nan
    ranks = _rankdata_with_ties(scores)
    sum_ranks_pos = ranks[labels==1].sum()
    U = sum_ranks_pos - n1*(n1+1)/2.0
    return U / (n1*n0)

# ---------- ROC/PR curves & AUPRC ----------
def roc_curve_from_scores(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n1 = labels.sum(); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return np.array([0,1]), np.array([0,1]), np.nan
    order = np.argsort(-scores)  # desc
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / n1
    fpr = fp / n0
    # prepend (0,0), append (1,1)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    # trapezoid area
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc

def pr_curve_from_scores(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n1 = labels.sum()
    if n1 == 0:
        return np.array([0,1]), np.array([1,0]), np.nan
    order = np.argsort(-scores)  # desc
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    recall    = tp / n1
    # anchor endpoints for pretty plotting
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])
    # step-wise interpolation
    auprc = np.trapz(precision, recall)
    return recall, precision, auprc

# ---------- SDT transforms ----------
def z_from_p(p):
    p = float(np.clip(p, 1e-12, 1-1e-12))
    return float(torch.distributions.Normal(0,1).icdf(torch.tensor(p)))

def dprime_from_auc(auc):
    if np.isnan(auc): return np.nan
    return math.sqrt(2.0) * z_from_p(auc)

def pooled_dprime(scores_pos, scores_neg):
    m1, m0 = float(np.mean(scores_pos)), float(np.mean(scores_neg))
    v1, v0 = float(np.var(scores_pos, ddof=1)), float(np.var(scores_neg, ddof=1))
    s = math.sqrt(max(1e-12, 0.5*(v1 + v0)))
    return (m1 - m0) / s

# ---------- block bootstrap over contiguous word blocks ----------
def block_bootstrap_auc(scores, labels, block_len=50, n_boot=500, seed=0):
    rng = random.Random(seed)
    N = len(scores)
    idx_blocks = [np.arange(i, min(N, i+block_len)) for i in range(0, N, block_len)]
    aucs = []
    for _ in range(n_boot):
        take = rng.choices(idx_blocks, k=len(idx_blocks))
        idx  = np.concatenate(take)
        yb   = labels[idx]
        sb   = scores[idx]
        # need both classes in the resample
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aucs.append(auroc(sb, yb))
    if not aucs:
        return np.nan, (np.nan, np.nan)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), (float(lo), float(hi))

# ---------- consensus utilities ----------
def consensus_counts(EB_by_rater, n_words):
    cnt = np.zeros(n_words, dtype=int)
    for EB_list in EB_by_rater.values():
        for w in EB_list:
            cnt[max(0, w-1)] += 1
    return cnt

def spearman_corr(x, y):
    # rank both (average ties), then Pearson
    rx = _rankdata_with_ties(np.asarray(x, float))
    ry = _rankdata_with_ties(np.asarray(y, float))
    rx = (rx - rx.mean())/ (rx.std(ddof=0)+1e-12)
    ry = (ry - ry.mean())/ (ry.std(ddof=0)+1e-12)
    return float(np.mean(rx*ry))

# ---------- optional: peak-finding for discrete EB decisions ----------
def find_peaks_nms(x, min_distance=5, min_height=None, min_prom=None):
    x = np.asarray(x, float); n = len(x)
    cand = [i for i in range(1, n-1) if x[i] > x[i-1] and x[i] >= x[i+1]]
    cand = np.array(cand, dtype=int)
    if min_height is not None:
        cand = cand[x[cand] >= min_height]
    if min_prom is not None and len(cand):
        win = max(3, min_distance)
        keep = []
        for i in cand:
            a,b = max(0,i-win), min(n-1,i+win)
            if x[i] - np.min(x[a:b+1]) >= min_prom:
                keep.append(i)
        cand = np.array(keep, dtype=int) if keep else np.array([], dtype=int)
    # NMS
    peaks, taken = [], np.zeros(n, dtype=bool)
    for idx in cand[np.argsort(-x[cand])]:
        if taken[idx]: continue
        peaks.append(idx)
        a,b = max(0, idx-min_distance), min(n-1, idx+min_distance)
        taken[a:b+1] = True
    peaks.sort()
    return np.array(peaks, dtype=int)

def match_with_tolerance(pred_idx, true_idx, tol=2):
    true = set(true_idx); used=set(); hits=0
    for p in pred_idx:
        cands = [t for t in true if t not in used and abs(t-p)<=tol]
        if cands:
            t = min(cands, key=lambda t: abs(t-p))
            used.add(t); hits += 1
    misses = len(true) - hits
    fas    = len(pred_idx) - hits
    return hits, misses, fas

def dprime_from_counts(hits, misses, fas, crs):
    H  = (hits + 0.5) / (hits + misses + 1.0)
    FA = (fas  + 0.5) / (fas  + crs    + 1.0)
    return z_from_p(H) - z_from_p(FA), H, FA


def plot_consensus_curve(scores, EB_by_rater, tol_words, title_tag, out_path):
    n_words = len(scores)
    # build consensus counts
    cnt = consensus_counts(EB_by_rater, n_words)
    uniq = sorted(set(cnt))
    xs, ys = [], []
    for thr in range(1, max(uniq)+1):
        y = build_labels_union(
            { "consensus": [i+1 for i,c in enumerate(cnt) if c>=thr] },  # pass EB list as after-word (shift +1)
            n_words, tol_words=tol_words
        )

        xs.append(thr)
        ys.append(auroc(scores, y))
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Consensus threshold (≥ #raters)"); plt.ylabel("AUROC")
    #plt.title(f"Consensus sensitivity — {title_tag}")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

    rho_all = spearman_corr(scores, cnt)
    return rho_all

# ========= Paper figure helpers: save both PNG & PDF, CI consensus, timeseries, captions =========

# ===================== Paper-ready plotting (drop-in) =====================

# Colorblind-safe (Okabe–Ito)
COL = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "grey":   "#8A8A8A",
}

def set_paper_style():
    plt.rcParams.update({
        # single-column Nature sizing ~85 mm; adjust as you like
        "figure.figsize": (3.35, 2.6),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.linestyle": "--",
        "grid.alpha": 0.22,
        "axes.grid": False,         # cleaner
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def _savefig_pair(pathbase: str):
    plt.tight_layout(pad=0.6)
    plt.savefig(f"{pathbase}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(f"{pathbase}.pdf",           bbox_inches="tight", transparent=True)
    plt.close()

def _prettify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(color="#333333", labelcolor="#222222")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

def _annot(ax, txt, loc=(0.03, 0.97)):
    ax.text(loc[0], loc[1], txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#222", bbox=dict(boxstyle="round,pad=0.18",
            facecolor="white", edgecolor="#BBBBBB", linewidth=0.6, alpha=0.9))

# -------- ROC + PR (with baseline line on PR, annotated stats) -------------
def plot_roc_pr(scores, labels, title_tag, out_prefix):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos_rate = float(labels.mean())

    # ROC
    fpr, tpr, auc = roc_curve_from_scores(scores, labels)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color=COL["blue"])
    ax.plot([0,1], [0,1], ls="--", lw=1.0, color=COL["grey"])
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0,1), ylim=(0,1))

    _annot(ax, f"AUC = {auc:.3f}",loc=(0.7, 0.1))
    _prettify_axes(ax)
    _savefig_pair(f"out/{out_prefix}_ROC")

    # PR
    rec, prec, auprc = pr_curve_from_scores(scores, labels)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, color=COL["blue"])
    ax.axhline(pos_rate, ls="--", lw=1.0, color=COL["grey"])
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1))
    _prettify_axes(ax)
    _annot(ax, f"AUPRC = {auprc:.3f}   baseline = {pos_rate:.3f}")
    _savefig_pair(f"out/{out_prefix}_PR")

    return float(auc), float(auprc)

# -------- Score distributions (tidy hist + medians) ------------------------
def plot_score_distributions(scores, labels, title_tag, out_path_base):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = scores[labels==1]
    neg = scores[labels==0]

    rng = np.quantile(scores, [0.002, 0.998])  # tight range
    bins = max(18, int(np.sqrt(len(scores))))

    fig, ax = plt.subplots()
    ax.hist(neg, bins=bins, range=rng, density=True, color=COL["blue"], alpha=0.35,
            edgecolor=COL["blue"], linewidth=0.7, label="non-EB")
    ax.hist(pos, bins=bins, range=rng, density=True, color=COL["orange"], alpha=0.35,
            edgecolor=COL["orange"], linewidth=0.7, label="EB")

    # Medians
    if len(pos): ax.axvline(np.median(pos), color=COL["orange"], lw=1.2)
    if len(neg): ax.axvline(np.median(neg), color=COL["blue"],   lw=1.2)

    ax.set(xlabel="log(prob)", ylabel="density")
    _prettify_axes(ax)
    ax.legend(frameon=False)
    _savefig_pair("out/"+out_path_base)

# -------- Consensus curve with 95% CIs + #positives (twin axis) ------------
def plot_consensus_curve_with_ci(scores, EB_by_rater, tol_words, title_tag, out_prefix,
                                 block_len=50, n_boot=2000, seed=0):
    n = len(scores)
    cnt = consensus_counts(EB_by_rater, n)
    K   = int(cnt.max())
    xs, aucs, pos_counts, y_by_thr = [], [], [], []
    for thr in range(1, K+1):
        pos_idx = [i for i,c in enumerate(cnt) if c >= thr]
        y_thr = np.zeros(n, dtype=np.int8); y_thr[pos_idx] = 1
        if tol_words > 0:
            y_thr = build_labels_union({"cons": [i+1 for i in pos_idx]}, n, tol_words=tol_words)
        xs.append(thr); y_by_thr.append(y_thr)
        aucs.append(auroc(scores, y_thr))
        pos_counts.append(int(y_thr.sum()))

    rng = random.Random(seed)
    blocks = [np.arange(i, min(n, i+block_len)) for i in range(0, n, block_len)]
    boot_lo = np.full(len(xs), np.nan); boot_hi = np.full(len(xs), np.nan)
    acc = [[] for _ in xs]
    for _ in range(n_boot):
        idx = np.concatenate(rng.choices(blocks, k=len(blocks)))
        s_b = np.asarray(scores)[idx]
        for j, y_thr in enumerate(y_by_thr):
            y_b = y_thr[idx]
            if y_b.sum()==0 or y_b.sum()==len(y_b): continue
            acc[j].append(auroc(s_b, y_b))
    for j in range(len(xs)):
        if acc[j]:
            boot_lo[j], boot_hi[j] = np.percentile(acc[j], [2.5, 97.5])

    fig, ax = plt.subplots()
    ax.errorbar(xs, aucs, yerr=[np.array(aucs)-boot_lo, boot_hi-np.array(aucs)],
                fmt="-o", color=COL["blue"], ecolor=COL["sky"], elinewidth=1.0, capsize=2.5)
    ax.set(xlabel="Consensus threshold (≥ #raters)", ylabel="AUROC", ylim=(min(0.9, np.nanmin(boot_lo)*0.995) if np.isfinite(boot_lo).any() else 0.9, 1.0))
    _prettify_axes(ax)

    ax2 = ax.twinx()
    ax2.bar(xs, pos_counts, color="#BBBBBB", alpha=0.25, width=0.6)
    ax2.set_ylabel("# positives")
    for x, y in zip(xs, pos_counts):
        ax2.text(x, y, str(y), ha="center", va="bottom", fontsize=7, color="#444")

    _savefig_pair(f"out/{out_prefix}_consensus_curve_CI")

    return {"thr": xs,
            "auc": [float(a) for a in aucs],
            "ci_lo": [float(x) for x in boot_lo],
            "ci_hi": [float(x) for x in boot_hi],
            "n_pos": [int(x) for x in pos_counts],
           }

# -------- Time-series with union EB + peaks + light smoothing --------------
def plot_timeseries_with_marks(ds, lp_word, EB_by_rater, peaks, title_tag, out_prefix):
    x = ds.data_times
    y = np.asarray(lp_word, float)

    # light smoothing (moving average 15 words)
    k = max(5, int(15))
    kern = np.ones(k) / k
    y_smooth = np.convolve(y, kern, mode="same")

    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    ax.plot(x, y, lw=0.8, color=COL["blue"], alpha=0.8, label="boundary score")
    ax.plot(x, y_smooth, lw=1.4, color=COL["red"], alpha=0.9, label="smoothed")

    union = build_labels_union(EB_by_rater, len(y), tol_words=0)
    idx = np.where(union==1)[0]
    if len(idx):
        ax.scatter(x[idx], y[idx], s=8, color=COL["green"], alpha=0.85, label="human EB (union)")
    if peaks is not None and len(peaks):
        ax.scatter(x[peaks], y[peaks], s=14, marker="P", color=COL["orange"], alpha=0.9, label="LLM peaks")

    ax.set(xlabel="time (s)", ylabel="score", title=f"Boundary score over time — {title_tag}")
    _prettify_axes(ax)
    ax.legend(frameon=False, ncols=3, loc="upper right")
    _savefig_pair(f"out/{out_prefix}_timeseries")


# ---------- punctuation helper ----------
_PUNCT_ENDERS = set(list("."))
_CLOSERS = set(list("])}»›”’\"'"))

def _strip_trailing_closers(token: str) -> str:
    t = token
    while len(t) and t[-1] in _CLOSERS:
        t = t[:-1]
    return t

def punctuation_flags(words: list[str]) -> np.ndarray:
    """
    Return a boolean array len(words) where True at word i if the word i
    ends with sentence punctuation (., !, ?, …, ;, :, ,) after stripping
    trailing quotes/closers. This marks punctuation *after* the word,
    which aligns with how lp_word is scored at boundaries after word i.
    """
    flags = np.zeros(len(words), dtype=bool)
    for i, w in enumerate(words):
        w2 = _strip_trailing_closers(w.strip())
        flags[i] = (len(w2) > 0) and (w2[-1] in _PUNCT_ENDERS)
    return flags



# ─────────────────────────────────────────────────────────────
# 3)  LOAD MODEL
# ─────────────────────────────────────────────────────────────


device = "cuda" if torch.cuda.is_available() else "cpu"

run_model_on_gpu = False
if run_model_on_gpu:
    model  = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
                cache_dir=os.getenv("HF_HOME"), quantization_config=bnb,
                low_cpu_mem_usage=False, do_sample=True,temperature=0.2).eval()

    print(f"Model ready — {vram()}",flush=True)

tok= AutoTokenizer.from_pretrained(
                MODEL_NAME, token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
                cache_dir=os.getenv("HF_HOME"))

#tok.add_special_tokens({"additional_special_tokens": [EVENT_MARKER]})
#model.resize_token_embeddings(len(tok),mean_resizing=False)
marker_id = tok(EVENT_MARKER, add_special_tokens=False).input_ids[0]
print("marker_id is =",marker_id,flush=True)
torch.set_grad_enabled(False)

# ─────────────────────────────────────────────────────────────
# 4)  STORIES
# ─────────────────────────────────────────────────────────────
stories = {"marot": marot}
only_newline = True          # skip heavy hidden‑state part


HR_dict ={
    "finn": """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte:¶ ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
¶Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. 
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.

Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte.
¶Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. ¶Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
¶Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“
¶Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging.
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
¶Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“
¶Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste.
¶Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
¶Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof.
¶Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.¶
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
¶Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.
¶Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
 ¶ Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
¶„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
¶Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden.¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“
¶Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.¶
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer.
¶Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
¶Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.¶
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.
¶Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben.
¶Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.""",
    "maddox": """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
  Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. ¶
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte. ¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. Die IT Abteilung hatte Rüeggs Mail Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“ ¶
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti Tunnel, der dampfende Mund eines Coffeeshops. ¶
Am Polyterrasse Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging. ¶
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“ ¶
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH Profilfoto, doch der Stoff war zerknittert vom Rastlos Sich Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste. ¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
Eine Spur führte sie über das Uni WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co Working Space beim Hauptbahnhof. ¶
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.
Der Klub residierte in einer Jugendstilvilla am Zürichberg.¶
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain Unternehmerin und Schweizer Blitzschach Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt in Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden. ¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold Wallet. IP Herkunft: der Co Working Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“ ¶
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer. ¶
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.
Die SD Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war. ¶
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben. ¶
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.
""",
    "yves": """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
  Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. 
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte. ¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“ ¶
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging. ¶
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“ ¶
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste. ¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof. ¶
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“. ¶
Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden. ¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“ ¶
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer. ¶
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett. ¶
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war. ¶
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben. ¶
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.""",
    "roya": """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
  Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.¶
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. 
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.¶
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte.¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“¶
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging.¶
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“¶
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“¶
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste.¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“¶
Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof.¶
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.¶
Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden.¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“¶
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“¶
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer.¶
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.¶
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.¶
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.¶
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben.¶
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.""",
    "leo": """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte. ¶

 

Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser. ¶

 

  Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.

Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern.

Der Gerichts mediziner kniete am Körper des Toten.

„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.

Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.

Ein eigenartiger Kontrapunkt zum regennassen Grau.

¶

Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.

Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte.

¶

Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.

Sie stellte sich unter die grelle Neonröhre: ¶

 

„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“

Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.

¶

 

Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ –

„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“

Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“

¶

Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“

¶

Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.

¶

Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging.

¶

 

Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.

Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.

„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“

„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.

Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“

Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.

Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.

„Wer hat Zugang?“

„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“

¶

Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.

Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.

„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.

„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“

¶

 

Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“

Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“

¶

 

Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste.

¶

 

Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.

Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:

  Rüeggs privates Ultrabook war weg.

In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.

„Notebook entwendet. Darauf die verschlüsselte Datenbank?“

Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“

¶

 

Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof.

Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.

Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.

¶

„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“

Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.

¶

 

Der Klub residierte in einer Jugendstilvilla am Zürichberg.

Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.

  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.

Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.

Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.

„Kommissarin? Welch seltene Gäste.“

Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“

Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“

„Was stand auf dem Spiel?“

„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“

Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden.

¶

Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.

„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“

Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.

Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“

¶

 

Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.

„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“

„Wieso?“

„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“

Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“

¶

Marot nickte Kern zu; er wählte lautlos die Einsatznummer.

¶

 

Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.

¶

Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.

„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“

Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“

Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.

 

Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.

Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.

 

Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben.

 

Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.

Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.

 

Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.""",
    "migi": """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
¶
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
¶
Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
¶
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. 
¶
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte.
¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. 
¶
Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. 
Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“
¶
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
¶
Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging.
¶
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste.
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof.
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.
¶
Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
 Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden.
¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“
¶
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer.
¶ 
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
¶
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. 
¶
Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben.
¶
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.""",
    "runa": """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
¶Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. ¶
Der Gerichtsmediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
¶Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich. 
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte. ¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp. 
Sie stellte sich unter die grelle Neonröhre: ¶
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an ¶„M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“
¶Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
¶Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, ¶bevor sie durch den Glasgang ins Herz der Hochschule ging.
¶Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“
¶Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel¶. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste. ¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof. ¶
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.¶
Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel. ¶
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden.
¶Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“
¶Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer.
¶Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.
¶Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.
¶Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben.
¶Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter. ¶
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.""",
    "nici": """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
  Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen. ¶
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. 
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau. 
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte. ¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH. ¶
Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging. ¶
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.¶
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste.¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof.
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.¶
Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden.¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer.
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.¶
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben.
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.""",
    "timo": """"Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
  Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. ¶
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte.¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.¶
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“¶
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“¶
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging.¶
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.¶
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“¶
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“¶
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. ¶ Marot ließ Maurer gehen – fürs Erste.¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“¶
Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof.¶
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“¶
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.¶
Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden.¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“¶
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“¶
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer.¶
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.¶
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.¶
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben.¶
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.¶
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug."""
}

marot_finn="""Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte:¶ ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
¶Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. 
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.

Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte.
¶Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. ¶Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
¶Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“
¶Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging.
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
¶Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“
¶Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste.
¶Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
¶Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof.
¶Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.¶
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
¶Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.
¶Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
 ¶ Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
¶„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
¶Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden.¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“
¶Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.¶
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer.
¶Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
¶Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.¶
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war.
¶Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben.
¶Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug."""
marot_maddox = """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
  Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. ¶
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte. ¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. Die IT Abteilung hatte Rüeggs Mail Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“ ¶
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti Tunnel, der dampfende Mund eines Coffeeshops. ¶
Am Polyterrasse Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging. ¶
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“ ¶
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH Profilfoto, doch der Stoff war zerknittert vom Rastlos Sich Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste. ¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
Eine Spur führte sie über das Uni WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co Working Space beim Hauptbahnhof. ¶
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“.
Der Klub residierte in einer Jugendstilvilla am Zürichberg.¶
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain Unternehmerin und Schweizer Blitzschach Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt in Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden. ¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold Wallet. IP Herkunft: der Co Working Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“ ¶
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer. ¶
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett.
Die SD Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war. ¶
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben. ¶
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug.
"""
marot_yves = """Der Regen hing schwer über Zürich, als Kommissarin Selina Marot an jenem bleigrauen Novembermorgen die Limmatbrücke überquerte.
Das Sirenen heulen der Spurensicherung brach sich an den Fassaden, und Nebelschleier krochen über das dunkle Wasser.
  Sie schob den Mantelkragen hoch, während ihre Gedanken bereits um den Fund kreisten, der sie aus dem warmen Büro gerissen hatte: ein unbekannter Mann, aufrecht in Todesstarre, mit einer roten Schachfigur in den gefalteten Händen.
Am Ufer roch es nach nassem Laub und Diesel. Blaulicht tauchte die Kastanienbäume in ein kaltes Flackern. 
Der Gerichts mediziner kniete am Körper des Toten.
„Etwa zwei Uhr morgens, präziser Stich unter das Brustbein“, murmelte er, ohne aufzuschauen.
Marot nahm die Schachfigur mit Pinzette auf: ein leuchtend roter König, Emaille auf Metall.
Ein eigenartiger Kontrapunkt zum regennassen Grau.
Weiter oben auf dem Pflaster verlor sich ein schmaler, rotbrauner Strich.
Keine Fußspuren, keine Reifenspuren, nur die ungeduldige Strömung der Limmat, die verriet, dass der Mörder Wasser als Verbündeten gewählt hatte. ¶
Das Büro des Rechtsmedizinischen Instituts lag nur zehn Gehminuten entfernt. Als Marot ihre Jacke abstreifte, liefen noch Tropfen vom Hutkremp.
Sie stellte sich unter die grelle Neonröhre:
„Fingerabdrücke?“ „Nichts Verwertbares – Handschuhe“, entgegnete der Kriminaltechniker. „Aber das Gesicht haben wir eingescannt.“
Die Computersoftware ratterte. Dann ploppte ein Name auf den Bildschirm: Dr. Felix Rüegg, 45, Neurowissenschaftler, ETH.
Ein Surren vibrierte in ihrer Manteltasche. Die IT‑Abteilung hatte Rüeggs Mail‑Account gespiegelt: Eine letzte Nachricht an „M.“ – 
„Ich habe die letzte Partie gewonnen. Morgen alles vorbei. 02:00 am Flussufer. Bring, was du mir schuldest.“
Marot lehnte sich zurück. „Partie gewonnen … Könnte Schach sein. Oder ein Deal.“
Sie seufzte, zog den Hut wieder auf. „Lassen Sie den Wagen vorfahren. Wir fahren zur ETH.“ ¶
Die Stadt rauschte an den Scheiben vorbei: Tramleitungen, Graffiti‑Tunnel, der dampfende Mund eines Coffeeshops.
Am Polyterrasse‑Parkhaus stiegen sie aus. Marot genoss einen Moment die klare Kälte, bevor sie durch den Glasgang ins Herz der Hochschule ging. ¶
Das Labor roch nach Metall und ozonigem Elektronikduft. Auf einem Whiteboard war ein Schachbrett skizziert, daneben verschachtelte Entscheidungs diagramme.
Eine junge Mitarbeiterin, Lynn Wu, trat ihnen entgegen, Nervosität in den Augen.
„Wir … wir haben erst heute Früh erfahren, dass Professor Rüegg tot ist.“
„Er forschte an ‚Project Gambit‘, korrekt?“ fragte Marot.
Wu nickte. „Ein KI‑System, die Börse mithilfe von Spieltheorie zu schlagen.“
Im Regal stand eine lackierte Figurensammlung – nur der rote König fehlte.
Marot strich mit dem Finger über den leeren Samt einsatz. Ein kalter Schauer kroch ihr den Rücken hinunter.
„Wer hat Zugang?“
„Nur wir Mitarbeitenden – und Dr. Maurer, sein Stellvertreter.“ ¶
Dr. Matthias Maurer erwartete sie im Vernehmungsraum des Polizeipräsidiums.
Er trug denselben grauen Pullover wie in seinem ETH‑Profilfoto, doch der Stoff war zerknittert vom Rastlos‑Sich‑Reiben.
„Professor Rüegg war besessen von seinem Projekt“, begann er und rieb die Hand innenseiten aneinander.
„Er hat in den letzten Wochen Daten manipuliert, um unsere Geldgeber bei Laune zu halten. Ich habe versucht, ihn zu stoppen.“
Marot legte ihr Diktiergerät zwischen sie. „Sie waren gestern Nacht an der Limmat.“
Maurer schüttelte heftig den Kopf. „Ich habe online Schach gespielt, Blitzturnier. Logs sind auf meinem Rechner.“
Kern, der Kriminaltechniker, überprüfte die Zeitstempel. Möglich, aber zu leicht fälschbar. Marot ließ Maurer gehen – fürs Erste. ¶
Noch am selben Abend suchten Marot und Kern Rüeggs Büro ein zweites Mal.
Die Schreibtischschubladen sperrten sich wie fest gebissene Zähne. Doch das, was fehlte, war auffälliger als jeder Fund:
  Rüeggs privates Ultrabook war weg.
In der Fensterbank lag nur eine einzelne Staubspur, quadratisch.
„Notebook entwendet. Darauf die verschlüsselte Datenbank?“
Kern nickte. „Und wahrscheinlich sein Tagebuch – er protokollierte jede ‚Partie‘.“
Eine Spur führte sie über das Uni‑WLAN – das Gerät hatte sich um 00:47 Uhr zuletzt eingeloggt, Standort: ein Co‑Working‑Space beim Hauptbahnhof. ¶
Der Hehlerladen im Kreis 4 war ein Kabinett aus Flimmer monitoren und verlöteten Platinen.
Zwischen Retro‑Konsolen lag tatsächlich ein dünnes Ultrabook, Typenbezeichnung identisch mit Rüeggs Dienstgerät, Gehäuse noch feucht vom Regen.
„Gestern Nacht reingekommen“, grunzte der Besitzer. „Eine junge Frau, asiatisch, Kapuzen mantel. Wollte schnell Cash.“
Kern löste die Rückwand: formatiert, aber im Lüftergitter klemmte eine winzige Papierfaser – das gestickte Wappen des exklusiven Schachklubs „KönigsGambit“. ¶
Der Klub residierte in einer Jugendstilvilla am Zürichberg.
Messingleuchter tauchten Parkett in warmes Gold; gedämpfter Jazz verschmolz mit dem Geräusch klackender Schachuhren.
  Marot ließ den Blick über die Bretter schweifen, bis ihr eine markant fröhliche Stimme auffiel.
Am Haupttisch spielte Marianne Hefti, Blockchain‑Unternehmerin und Schweizer Blitzschach‑Meisterin.
Sie beendete gerade eine Partie mit einem kühlen Matt‑in‑Drei und drehte sich lächelnd um.
„Kommissarin? Welch seltene Gäste.“
Marot zeigte ihr den roten König. „Rüegg hatte ihn in den Händen, als er starb.“
Hefti legte den Kopf schief. „Er gewann gestern eine Armageddon‑Runde gegen mich. Gute Partie, zugegeben. Danach verschwand er zum Treffen am Fluss.“
„Was stand auf dem Spiel?“
„Sein Algorithmus gegen meine Investorenkontakte. Er behielt die Oberhand, also schlug er den König.“
Metaphorisch – oder wörtlich? Marot konnte noch nicht entscheiden. ¶
Zurück im Präsidium, während der Himmel in ein rußiges Violett überging, starrte Kern auf endlose Spalten von Krypto transaktionen.
„Gestern 23:43 Uhr: 200 000 US‑Stablecoins auf Rüeggs Wallet“, sagte er. „Um 00:15 Uhr raus an ein Cold‑Wallet. IP‑Herkunft: der Co‑Working‑Space.“
Eine Kamera aufnahme zeigte Lynn Wu, wie sie das Gebäude betrat, den Laptop eng an die Brust gepresst.
Marot strich sich übers Kinn. „Sie ist tiefer drin, als sie zugibt.“ ¶
Sie fanden Wu in ihrer Altbauwohnung, zwischen rauschenden Server türmen. Auf dem Parkett verstreut lagen ausgedruckte Code‑Schnipsel und ein Schachbrett mit laufender Position. Wu hob die Hände, als Marot eintrat.
„Ich habe ihn nicht getötet“, stammelte sie. „Ich war spät dran … Felix lag schon dort, Blut überall. Ich hab nur die Datenkarte genommen.“
„Wieso?“
„Weil sie uns alle zerstört hätte. Project Gambit war mein Leben, mein Visum, alles.“
Wu zitterte. „Die Karte steckt hinter dem Porträt des ersten Klubpräsidenten. Ich wollte sie heute verschwinden lassen.“
Marot nickte Kern zu; er wählte lautlos die Einsatznummer. ¶
Die Villa wirkte noch prächtiger im strömenden Regen, als das SEK anrückte. Unter dem Knarren antiker Dielen hallten Befehle, ehe Stille sich senkte wie ein Tuch. Hinter dem staubigen Porträt fand sich tatsächlich die metallene SD‑Kartenhülle.
Doch plötzlich hallte das Klacken einer entsicherten Waffe. Dr. Maurer trat aus den Schatten des Rauchersalons, Pistole im zitternden Griff.
„Rühr dich nicht, Selina“, hauchte er. Tränen strömten ihm übers Gesicht. „Felix wollte mich demütigen, mir alles nehmen. Ich war das Genie, nicht er!“
Marot breitete die Handflächen. „Matthias, deine Tochter wird dich noch sehen wollen. Senk die Waffe. Wir beenden diese Partie ohne weiteres Blut.“
Maurers Blick flackerte, als ob er innerlich noch einmal alle Züge durchrechnete. Dann sanken seine Schultern, Metall klirrte auf Parkett. ¶
Die SD‑Karte enthielt Rüeggs vollständiges Tagebuch: genaue Log‑Dateien über Maurers Datentricks, Finanzbetrug, Drohungen.
Die rote Schachfigur tauchte immer wieder als Markierung auf – ein Symbol für den Augenblick, in dem der Gegner „königslos“ war. ¶
Vor Gericht gestand Maurer den Mord; ein kalkulierter Stich, als Rüegg ihn erpressen wollte. Lynn Wu erhielt Bewährungs strafe wegen Hehlerei und Datenverbergung, Marianne Hefti kam glimpflich davon. Die ETH stoppte Project Gambit und rief eine Ethik kommission ins Leben. ¶
Einige Wochen später lag frostiger Dunst über der Limmat. Marot stützte die Ellbogen auf das Brückengeländer, dieselbe Stelle, an der alles begonnen hatte. Aus dem Polizeifunk in ihrer Manteltasche knackte eine neue Meldung.
Sie holte den roten König aus der Tasche, drehte ihn im Licht des frühen Morgens. Dann steckte sie ihn zurück, wandte sich ab und stieg die Stufen zum Quai hinunter.
Der Regen von Zürich hatte sich gelegt, doch in den Köpfen der Menschen klapperten noch immer Schachuhren – und irgendwo plante schon jemand den nächsten Zug."""

maupassant_finn = """Guy de Maupassant Die Hand. ¶ Man drängte sich um den Untersuchungsrichter Bermutier, der seine Ansicht äußerte über den mysteriösen Fall in Saint Cloud. ¶Seit einem Monat entsetzte dies unerklärliche Verbrechen Paris. Niemand konnte es erklären. ¶Herr Bermutier stand, den Rücken gegen den Kamin gelehnt da, sprach, sichtete die Beweisstücke, kritisierte die verschiedenen Ansichten darüber, aber er selbst gab kein Urteil ab. Ein paar Damen waren aufgestanden, um näher zu sein, blieben vor ihm stehen, indem sie an den glattrasierten Lippen des Beamten hingen, denen so ernste Worte entströmten. Sie zitterten und schauerten ein wenig zusammen in neugieriger Angst und dem glühenden unersättlichen Wunsch nach Grauenhaftem, der ihre Seelen quälte und peinigte. Eine von ihnen, bleicher als die anderen, sagte während eines Augenblicks Stillschweigen : – Das ist ja schrecklich! Es ist wie etwas Übernatürliches dabei. Man wird die Wahrheit nie erfahren. Der Beamte wandte sich zu ihr : – Ja, gnädige Frau, wahrscheinlich wird man es nicht erfahren, aber wenn Sie von Übernatürlichem sprechen, so ist davon nicht die Rede. Wir stehen vor einem sehr geschickt ausgedachten und ungemein geschickt ausgeführten Verbrechen, das so mit dem Schleier des Rätselhaften umhüllt ist, daß wir die unbekannten Nebenumstände nicht zu entschleiern vermögen. Aber ich habe früher einmal selbst einen ähnlichen Fall zu bearbeiten gehabt, in den sich auch etwas Phantastisches zu mischen schien. Übrigens mußte man das Verfahren einstellen, da man der Sache nicht auf die Spur kam. Mehrere Damen sagten zu gleicher Zeit, so schnell, daß ihre Stimmen zusammenklangen : – Ach Gott, erzählen Sie uns das! Der Beamte lächelte ernst, wie ein Untersuchungsrichter lächeln muß, und sagte : – Glauben Sie ja nicht, daß ich auch nur einen Augenblick gemeint habe, bei der Sache wäre etwas Übernatürliches. Es geht meiner Ansicht nach alles mit rechten Dingen zu. Aber wenn sie statt ›übernatürlich‹ für das was wir nicht verstehen, einfach ›unaufklärbar‹ sagen, so wäre das viel besser. Jedenfalls interessierten mich bei dem Fall, den ich Ihnen erzählen werde, mehr die Nebenumstände. Es handelte sich etwa um folgendes : ¶ Ich war damals Untersuchungsrichter in Ajaccio, einer kleinen weißen Stadt an einem wundervollen Golf, der rings von hohen Bergen umstanden ist. Ich hatte dort hauptsächlich Vendetta - Fälle zu verfolgen. ¶Es giebt wundervolle, so tragisch wie nur möglich, wild und leidenschaftlich. Dort kommen die schönsten Rächerakte vor, die man sich nur träumen kann, Jahrhunderte alter Haß, nur etwas verblaßt, aber nie erloschen. Unglaubliche Listen, Mordfälle, die zu wahren Massakren, sogar beinahe zu herrlichen Thaten ausarten. ¶Seit zwei Jahren hörte ich nur immer von der Blutrache, diesem furchtbaren, korsischen Vorurteil, das die Menschen zwingt, Beleidigungen nicht bloß an der Person, die sie gethan, zu rächen, sondern auch an den Kindern und Verwandten. Ich hatte ihm Greise, Kinder, Vettern zum Opfer fallen sehen, ich steckte ganz voll solcher Geschichten. ¶ Da erfuhr ich eines Tages, daß ein Engländer auf mehrere Jahre eine im Hintergrund des Golfes gelegene Villa gemietet. Er hatte einen französischen Diener mitgebracht, den er in Marseille gemietet. Bald sprach alle Welt von diesem merkwürdigen Manne, der in dem Haus allein lebte und nur zu Jagd und Fischfang ausging. Er redete mit niemand, kam nie in die Stadt, und jeden Morgen übte er sich ein oder zwei Stunden im Pistolen - oder Karabiner - Schießen. Allerlei Legenden bildeten sich um den Mann. Es wurde behauptet, er wäre eine vornehme Persönlichkeit, die aus politischen Gründen aus seinem Vaterlande entflohen. Dann ging das Gerücht, daß er sich nach einem furchtbaren Verbrechen hier versteckt hielt ; man erzählte sogar grauenvolle Einzelheiten. ¶Ich wollte in meiner Eigenschaft als Untersuchungsrichter etwas über den Mann erfahren, aber es war mir nicht möglich. Er ließ sich Sir John Rowell nennen. Ich begnügte mich also damit, ihn näher zu beobachten, und ich kann nur sagen, daß man mir nichts irgendwie Verdächtiges mitteilen konnte. Aber da die Gerüchte über ihn fortgingen, immer seltsamer wurden und sich immer mehr verbreiteten, so entschloß ich mich, einmal den Fremden selbst zu sehen, und ich begann regelmäßig in der Nähe seines Besitztums auf die Jagd zu gehen. Ich wartete lange auf eine Gelegenheit. ¶ Endlich bot sie sich mir dadurch, daß ich dem Engländer ein Rebhuhn vor der Nase wegschoß. Mein Hund brachte es mir, ich nahm es auf, entschuldigte mich Sir John Rowell gegenüber und bat ihn artig, die Beute anzunehmen. ¶Er war ein großer, rothaariger Mann, mit rotem Bart, sehr breit und kräftig, eine Art ruhiger, höflicher Herkules. Er hatte nichts von der sprüchwörtlichen englischen Steifheit und dankte mir lebhaft für meine Aufmerksamkeit in einem englisch gefärbten Französisch. ¶ Nach vier Wochen hatten wir fünf oder sechs Mal zusammen gesprochen, und ¶ eines Abends, als ich an seiner Thür vorüberkam, sah ich ihn, wie er in seinem Garten rittlings auf einem Stuhl saß und die Pfeife rauchte. Ich grüßte, und er lud mich zu einem Glase Bier ein. Das ließ ich mir nicht zweimal sagen. Er empfing mich mit aller peinlichen englischen Artigkeit, sprach am höchsten Lobeston von Frankreich, von Korsika, und erklärte, er hätte dieses Eiland zu gern. ¶ Da stellte ich ihm mit größter Vorsicht, indem ich lebhaftes Interesse heuchelte, einige Fragen über sein Leben und über seine Absichten. Ohne Verlegenheit antwortete er mir, erzählte mir, er sei sehr viel gereist, in Afrika, Indien und Amerika und fügte lachend hinzu : – O, ich haben viele Abenteuer gehabt, o yes! ¶ Dann sprach ich weiter von der Jagd, und er erzählte mir interessante Einzelheiten über die Nilpferd -, Tiger -, Elephanten - und sogar Gorilla - Jagd. Ich sagte : – Alle diese Tiere sind gefährlich! Er lächelte : – O no, die schlimmste ist die Mensch! Er lachte gemütlich, in seiner behäbigen englischen Art und sagte : – Ich habe auch viel die Mensch gejagt! ¶Dann sprach er von Waffen und forderte mich auf, bei ihm einzutreten, um ein paar Gewehre verschiedener Systeme zu besehen. Das Wohnzimmer war mit schwarzer, gestickter Seide ausgeschlagen, große, gelbe Blumen schlängelten sich über den dunklen Stoff und leuchteten wie Feuer. Er sagte : – Das ist japanische Stickerei! Aber mitten auf der größten Wand zog ein eigentümlicher Gegenstand meine Blicke auf sich. Von vier Ecken mit rotem Sammet umgeben, hob sich etwas Seltsames ab. Ich trat näher. Es war eine Hand. Eine menschliche Hand. Nicht die Hand eines Skelettes mit gebleichten, reinlich präparierten Knochen, sondern eine schwarze, vertrocknete Hand mit gelben Nägeln, bloßliegenden Muskeln und alten Blutspuren von dem glatt abgeschnittenen Knochen, als wäre er mitten im Unterarm mit einem Beile abgehackt. An dem Handgelenk war eine Riesen - Eisenkette befestigt, die mit einem so starken Ring, als wolle man einen Elephant daran binden, die Hand an der Mauer hielt. ¶ Ich fragte : – Was ist denn das? Der Engländer antwortete ganz ruhig : – ¶ Das war meine beste Feind ; sie kam von Amerika. Das ist mit die Säbel abgeschlagen und die Haut mit scharfe Kiesel abgekratzt und acht Tage in die Sonne getrocknet. Aho, sehr fein für mir! ¶ Ich faßte diese menschlichen Überreste, die einem Koloß angehört haben mußten, an. Diese Hand war gräßlich zu sehen, und unwillkürlich drängte sich mir der Gedanke an einen fürchterlichen Racheakt auf. Ich sagte : – Dieser Mann muß sehr stark gewesen sein! Der Engländer antworte ganz weich : – O yes, aber ich war stärker, ich hatte die Kette angebunden, sie zu halten. Ich meinte, er scherze und sagte : – Nun, diese Kette ist ja jetzt unnütz, die Hand wird ja nicht davon laufen. Sir John Rowell antwortete ernst : – Er wollte immer fortlaufen, die Kette war nötig. ¶Mein Blick ruhte fragend auf seinem Gesicht, und ich sagte mir : Ist der Kerl verrückt, oder ist es ein schlechter Witz? Aber sein Gesicht blieb unbeweglich ruhig, voller Wohlwollen, er sprach von anderen Dingen, und ich bewunderte seine Gewehre. Aber ich bemerkte, daß geladene Revolver hier und da auf den Tischen lagen, als ob er in ständiger Furcht vor einem Angriff lebte. ¶ Ich besuchte ihn noch ein paar Mal, dann nicht mehr, man hatte sich an seine Anwesenheit gewöhnt, er war uns allen uninteressant geworden. ¶ Ein ganzes Jahr verstrich, da weckte mich eines Morgens, Ende September, mein Diener mit der Meldung, Sir John Rowell wäre in der Nacht ermordet worden. ¶ Eine halbe Stunde später betrat ich mit dem Gendarmerie - Hauptmann das Haus des Engländers. Der Diener stand ganz verzweifelt vor der Thür und weinte. Ich hatte zuerst den Mann in Verdacht, aber er war unschuldig. ¶Den Schuldigen hat man nie entdecken können. ¶ Als ich in das Wohnzimmer des Sir John Rowell. trat, sah ich auf den ersten Blick mitten in dem Raum die Leiche auf dem Rücken liegen. Die Weste war zerrissen, ein Ärmel hing herab, alles deutete darauf hin, daß ein furchtbarer Kampf stattgefunden hatte. Der Engländer war erwürgt worden, sein schwarzes, gedunsenes Gesicht hatte etwas Gräßliches und schien ein furchtbares Entsetzen auszudrücken. Zwischen den zusammengebissenen Zähnen steckte etwas und sein blutiger Hals war von fünf Löchern durchbohrt, als wären fünf Eisenspitzen dort eingedrungen. ¶ Ein Arzt folgte uns, er betrachtete lange die Fingerspuren im Fleisch und that die seltsame Äußerung : – Das ist ja, als ob er von einem Skelett erwürgt worden wäre. ¶ Ein Schauder lief mir über den Rücken, und ich blickte zur Wand, auf die Stelle, wo ich sonst die entsetzliche Hand gesehen. Sie war nicht mehr da, die Kette hing zerbrochen herab. Da beugte ich mich zu dem Toten nieder und fand in seinem verzerrten Mund einen der Finger dieser verschwundenen Hand. Gerade am zweiten Glied von den Zähnen abgebissen, oder vielmehr abgesägt. ¶Die Untersuchung wurde eingeleitet, man fand nichts, keine Thür war aufgebrochen worden, kein Fenster, kein Möbel. Die beiden Wachthunde waren nicht wach geworden. Die Aussage des Dieners war etwa folgende : ¶ Seit einem Monat schien sein Herr sehr erregt, er hatte viele Briefe bekommen, aber sie sofort wieder verbrannt. Oft nahm er in einem Wutanfall, fast tobsuchtartig, eine Reitpeische und schlug ein auf diese vertrocknete Hand, die an die Mauer geschmiedet und, man weiß nicht wie, zur Stunde, als das Verbrechen geschehen, geraubt worden war. Er ging sehr spät zu Bett und schloß sich jedesmal sorgfältig ein. Er hatte immer Waffen bei der Hand, manchmal sprach er Nachts laut, als zankte er sich mit jemandem. Diese Nacht hatte er aber zufällig keinen Lärm gemacht, und der Diener hatte Sir John erst ermordet vorgefunden, als er die Fenster öffnete. Er hatte niemandem im Verdacht. ¶ Was ich wußte, teilte ich dem Beamten und der Polizei mit, und auf der ganzen Insel wurde sorgfältig nachgeforscht – man entdeckte nichts. ¶Da hatte ich eine Nacht, ein Vierteljahr nach dem Verbrechen, einen furchtbaren Traum. Es war mir, als sähe ich die Hand, die entsetzliche Hand wie einen Skorpion, wie eine Spinne längs der Vorhänge hinhuschen. Dreimal wachte ich auf, dreimal schlief ich wieder ein, dreimal sah ich dieses entsetzliche Überbleibsel um mein Zimmer herumjagen, indem es die Finger wie Pfoten bewegte. ¶Am nächsten Tage brachte man mir die Hand, die man auf dem Kirchhof, wo Sir John Rowell begraben war, da man seine Familie nicht eruiert hatte, auf seinem Grabe gefunden hatte. Der Zeigefinger fehlte. ¶ Das, meine Damen, ist meine Geschichte, mehr weiß ich nicht. ¶Die Damen waren bleich geworden, zitterten, und eine von ihnen rief : – Aber das ist doch keine Lösung und keine Erklärung, wir können ja garnicht schlafen, wenn Sie uns nicht sagen, was Ihrer Ansicht nach passiert ist. Der Beamte lächelte ernst : – O meine Damen, ich will Sie gewiß nicht um Ihre schönsten Träume bringen, ich denke ganz einfach, daß der Besitzer dieser Hand gar nicht tot war und daß er einfach gekommen ist, um sie mit der Hand wieder zu holen, die ihm übrig geblieben war ; aber ich weiß nicht, wie er das angestellt hat. Das wird eine Art Vendetta sein. Eine der Damen flüsterte : – Nein, das kann nicht so gewesen sein! Und der Untersuchungsrichter schloß immer noch lächelnd : – Ich habe es Ihnen doch gesagt, daß meine Erklärung Ihnen nicht passen würde."""
maupassant_davide = """Guy de Maupassant Die Hand.¶ Man drängte sich um den Untersuchungsrichter Bermutier, der seine Ansicht äußerte über den mysteriösen Fall in Saint Cloud. ¶Seit einem Monat entsetzte dies unerklärliche Verbrechen Paris. Niemand konnte es erklären.¶ Herr Bermutier stand, den Rücken gegen den Kamin gelehnt da, sprach, sichtete die Beweisstücke, kritisierte die verschiedenen Ansichten darüber, aber er selbst gab kein Urteil ab.¶ Ein paar Damen waren aufgestanden, um näher zu sein, blieben vor ihm stehen, indem sie an den glattrasierten Lippen des Beamten hingen, denen so ernste Worte entströmten. Sie zitterten und schauerten ein wenig zusammen in neugieriger Angst und dem glühenden unersättlichen Wunsch nach Grauenhaftem, der ihre Seelen quälte und peinigte. Eine von ihnen, bleicher als die anderen, sagte während eines Augenblicks Stillschweigen : – Das ist ja schrecklich! Es ist wie etwas Übernatürliches dabei. Man wird die Wahrheit nie erfahren. Der Beamte wandte sich zu ihr : – Ja, gnädige Frau, wahrscheinlich wird man es nicht erfahren, aber wenn Sie von Übernatürlichem sprechen, so ist davon nicht die Rede. Wir stehen vor einem sehr geschickt ausgedachten und ungemein geschickt ausgeführten Verbrechen, das so mit dem Schleier des Rätselhaften umhüllt ist, daß wir die unbekannten Nebenumstände nicht zu entschleiern vermögen. Aber ich habe früher einmal selbst einen ähnlichen Fall zu bearbeiten gehabt, in den sich auch etwas Phantastisches zu mischen schien. Übrigens mußte man das Verfahren einstellen, da man der Sache nicht auf die Spur kam. Mehrere Damen sagten zu gleicher Zeit, so schnell, daß ihre Stimmen zusammenklangen : – Ach Gott, erzählen Sie uns das! Der Beamte lächelte ernst, wie ein Untersuchungsrichter lächeln muß, und sagte : – Glauben Sie ja nicht, daß ich auch nur einen Augenblick gemeint habe, bei der Sache wäre etwas Übernatürliches. Es geht meiner Ansicht nach alles mit rechten Dingen zu. Aber wenn sie statt ›übernatürlich‹ für das was wir nicht verstehen, einfach ›unaufklärbar‹ sagen, so wäre das viel besser. Jedenfalls interessierten mich bei dem Fall, den ich Ihnen erzählen werde, mehr die Nebenumstände. Es handelte sich etwa um folgendes : ¶Ich war damals Untersuchungsrichter in Ajaccio, einer kleinen weißen Stadt an einem wundervollen Golf, der rings von hohen Bergen umstanden ist. Ich hatte dort hauptsächlich Vendetta - Fälle zu verfolgen. Es giebt wundervolle, so tragisch wie nur möglich, wild und leidenschaftlich. Dort kommen die schönsten Rächerakte vor, die man sich nur träumen kann, Jahrhunderte alter Haß, nur etwas verblaßt, aber nie erloschen. Unglaubliche Listen, Mordfälle, die zu wahren Massakren, sogar beinahe zu herrlichen Thaten ausarten. Seit zwei Jahren hörte ich nur immer von der Blutrache, diesem furchtbaren, korsischen Vorurteil, das die Menschen zwingt, Beleidigungen nicht bloß an der Person, die sie gethan, zu rächen, sondern auch an den Kindern und Verwandten. Ich hatte ihm Greise, Kinder, Vettern zum Opfer fallen sehen, ich steckte ganz voll solcher Geschichten. ¶Da erfuhr ich eines Tages, daß ein Engländer auf mehrere Jahre eine im Hintergrund des Golfes gelegene Villa gemietet. Er hatte einen französischen Diener mitgebracht, den er in Marseille gemietet.¶ Bald sprach alle Welt von diesem merkwürdigen Manne, der in dem Haus allein lebte und nur zu Jagd und Fischfang ausging. Er redete mit niemand, kam nie in die Stadt, und jeden Morgen übte er sich ein oder zwei Stunden im Pistolen - oder Karabiner - Schießen. ¶Allerlei Legenden bildeten sich um den Mann. Es wurde behauptet, er wäre eine vornehme Persönlichkeit, die aus politischen Gründen aus seinem Vaterlande entflohen. Dann ging das Gerücht, daß er sich nach einem furchtbaren Verbrechen hier versteckt hielt ; man erzählte sogar grauenvolle Einzelheiten. Ich wollte in meiner Eigenschaft als Untersuchungsrichter etwas über den Mann erfahren, aber es war mir nicht möglich. Er ließ sich Sir John Rowell nennen. Ich begnügte mich also damit, ihn näher zu beobachten, und ich kann nur sagen, daß man mir nichts irgendwie Verdächtiges mitteilen konnte. Aber da die Gerüchte über ihn fortgingen, immer seltsamer wurden und sich immer mehr verbreiteten, so entschloß ich mich, einmal den Fremden selbst zu sehen, und ich begann regelmäßig in der Nähe seines Besitztums auf die Jagd zu gehen. Ich wartete lange auf eine Gelegenheit. ¶Endlich bot sie sich mir dadurch, daß ich dem Engländer ein Rebhuhn vor der Nase wegschoß. Mein Hund brachte es mir, ich nahm es auf, entschuldigte mich Sir John Rowell gegenüber und bat ihn artig, die Beute anzunehmen. Er war ein großer, rothaariger Mann, mit rotem Bart, sehr breit und kräftig, eine Art ruhiger, höflicher Herkules. Er hatte nichts von der sprüchwörtlichen englischen Steifheit und dankte mir lebhaft für meine Aufmerksamkeit in einem englisch gefärbten Französisch.¶ Nach vier Wochen hatten wir fünf oder sechs Mal zusammen gesprochen, ¶und eines Abends, als ich an seiner Thür vorüberkam, sah ich ihn, wie er in seinem Garten rittlings auf einem Stuhl saß und die Pfeife rauchte.¶ Ich grüßte, und er lud mich zu einem Glase Bier ein. Das ließ ich mir nicht zweimal sagen. Er empfing mich mit aller peinlichen englischen Artigkeit, sprach am höchsten Lobeston von Frankreich, von Korsika, und erklärte, er hätte dieses Eiland zu gern. Da stellte ich ihm mit größter Vorsicht, indem ich lebhaftes Interesse heuchelte, einige Fragen über sein Leben und über seine Absichten. Ohne Verlegenheit antwortete er mir, erzählte mir, er sei sehr viel gereist, in Afrika, Indien und Amerika und fügte lachend hinzu : – O, ich haben viele Abenteuer gehabt, o yes! Dann sprach ich weiter von der Jagd, und er erzählte mir interessante Einzelheiten über die Nilpferd -, Tiger -, Elephanten - und sogar Gorilla - Jagd. Ich sagte : – Alle diese Tiere sind gefährlich! Er lächelte : – O no, die schlimmste ist die Mensch! Er lachte gemütlich, in seiner behäbigen englischen Art und sagte : – Ich habe auch viel die Mensch gejagt! Dann sprach er von Waffen und forderte mich auf, bei ihm einzutreten, um ein paar Gewehre verschiedener Systeme zu besehen.¶ Das Wohnzimmer war mit schwarzer, gestickter Seide ausgeschlagen, große, gelbe Blumen schlängelten sich über den dunklen Stoff und leuchteten wie Feuer. Er sagte : – Das ist japanische Stickerei! ¶Aber mitten auf der größten Wand zog ein eigentümlicher Gegenstand meine Blicke auf sich. Von vier Ecken mit rotem Sammet umgeben, hob sich etwas Seltsames ab. Ich trat näher. Es war eine Hand. Eine menschliche Hand. Nicht die Hand eines Skelettes mit gebleichten, reinlich präparierten Knochen, sondern eine schwarze, vertrocknete Hand mit gelben Nägeln, bloßliegenden Muskeln und alten Blutspuren von dem glatt abgeschnittenen Knochen, als wäre er mitten im Unterarm mit einem Beile abgehackt. An dem Handgelenk war eine Riesen - Eisenkette befestigt, die mit einem so starken Ring, als wolle man einen Elephant daran binden, die Hand an der Mauer hielt. Ich fragte : – Was ist denn das? Der Engländer antwortete ganz ruhig : – Das war meine beste Feind ; sie kam von Amerika. Das ist mit die Säbel abgeschlagen und die Haut mit scharfe Kiesel abgekratzt und acht Tage in die Sonne getrocknet. Aho, sehr fein für mir! Ich faßte diese menschlichen Überreste, die einem Koloß angehört haben mußten, an. Diese Hand war gräßlich zu sehen, und unwillkürlich drängte sich mir der Gedanke an einen fürchterlichen Racheakt auf. Ich sagte : – Dieser Mann muß sehr stark gewesen sein! Der Engländer antworte ganz weich : – O yes, aber ich war stärker, ich hatte die Kette angebunden, sie zu halten. Ich meinte, er scherze und sagte : – Nun, diese Kette ist ja jetzt unnütz, die Hand wird ja nicht davon laufen. Sir John Rowell antwortete ernst : – Er wollte immer fortlaufen, die Kette war nötig.¶ Mein Blick ruhte fragend auf seinem Gesicht, und ich sagte mir : Ist der Kerl verrückt, oder ist es ein schlechter Witz? Aber sein Gesicht blieb unbeweglich ruhig, voller Wohlwollen,¶ er sprach von anderen Dingen, und ich bewunderte seine Gewehre. Aber ich bemerkte, daß geladene Revolver hier und da auf den Tischen lagen, als ob er in ständiger Furcht vor einem Angriff lebte.¶ Ich besuchte ihn noch ein paar Mal, dann nicht mehr, man hatte sich an seine Anwesenheit gewöhnt, er war uns allen uninteressant geworden.¶ Ein ganzes Jahr verstrich,¶ da weckte mich eines Morgens, Ende September, mein Diener mit der Meldung, Sir John Rowell wäre in der Nacht ermordet worden. ¶Eine halbe Stunde später betrat ich mit dem Gendarmerie - Hauptmann das Haus des Engländers.¶ Der Diener stand ganz verzweifelt vor der Thür und weinte. Ich hatte zuerst den Mann in Verdacht, aber er war unschuldig.¶ Den Schuldigen hat man nie entdecken können.¶ Als ich in das Wohnzimmer des Sir John Rowell. trat, sah ich auf den ersten Blick mitten in dem Raum die Leiche auf dem Rücken liegen. Die Weste war zerrissen, ein Ärmel hing herab, alles deutete darauf hin, daß ein furchtbarer Kampf stattgefunden hatte. Der Engländer war erwürgt worden, sein schwarzes, gedunsenes Gesicht hatte etwas Gräßliches und schien ein furchtbares Entsetzen auszudrücken. Zwischen den zusammengebissenen Zähnen steckte etwas und sein blutiger Hals war von fünf Löchern durchbohrt, als wären fünf Eisenspitzen dort eingedrungen. Ein Arzt folgte uns, er betrachtete lange die Fingerspuren im Fleisch und that die seltsame Äußerung : – Das ist ja, als ob er von einem Skelett erwürgt worden wäre. Ein Schauder lief mir über den Rücken, und ich blickte zur Wand, auf die Stelle, wo ich sonst die entsetzliche Hand gesehen. Sie war nicht mehr da, die Kette hing zerbrochen herab. Da beugte ich mich zu dem Toten nieder und fand in seinem verzerrten Mund einen der Finger dieser verschwundenen Hand. Gerade am zweiten Glied von den Zähnen abgebissen, oder vielmehr abgesägt.¶ Die Untersuchung wurde eingeleitet, man fand nichts, keine Thür war aufgebrochen worden, kein Fenster, kein Möbel. Die beiden Wachthunde waren nicht wach geworden. Die Aussage des Dieners war etwa folgende :¶ Seit einem Monat schien sein Herr sehr erregt, er hatte viele Briefe bekommen, aber sie sofort wieder verbrannt. Oft nahm er in einem Wutanfall, fast tobsuchtartig, eine Reitpeische und schlug ein auf diese vertrocknete Hand, die an die Mauer geschmiedet und, man weiß nicht wie, ¶zur Stunde, als das Verbrechen geschehen, geraubt worden war.¶ Er ging sehr spät zu Bett und schloß sich jedesmal sorgfältig ein. Er hatte immer Waffen bei der Hand, manchmal sprach er Nachts laut, als zankte er sich mit jemandem.¶ Diese Nacht hatte er aber zufällig keinen Lärm gemacht, und der Diener hatte Sir John erst ermordet vorgefunden, als er die Fenster öffnete. Er hatte niemandem im Verdacht. ¶Was ich wußte, teilte ich dem Beamten und der Polizei mit, ¶und auf der ganzen Insel wurde sorgfältig nachgeforscht – man entdeckte nichts. Da hatte ich eine Nacht, ¶ein Vierteljahr nach dem Verbrechen, einen furchtbaren Traum. Es war mir, als sähe ich die Hand, die entsetzliche Hand wie einen Skorpion, wie eine Spinne längs der Vorhänge hinhuschen. Dreimal wachte ich auf, dreimal schlief ich wieder ein, dreimal sah ich dieses entsetzliche Überbleibsel um mein Zimmer herumjagen, indem es die Finger wie Pfoten bewegte.¶ Am nächsten Tage brachte man mir die Hand, die man auf dem Kirchhof, wo Sir John Rowell begraben war, da man seine Familie nicht eruiert hatte, auf seinem Grabe gefunden hatte. Der Zeigefinger fehlte.¶ Das, meine Damen, ist meine Geschichte, mehr weiß ich nicht.¶ Die Damen waren bleich geworden, zitterten, und eine von ihnen rief : – Aber das ist doch keine Lösung und keine Erklärung, wir können ja garnicht schlafen, wenn Sie uns nicht sagen, was Ihrer Ansicht nach passiert ist. Der Beamte lächelte ernst : – O meine Damen, ich will Sie gewiß nicht um Ihre schönsten Träume bringen, ich denke ganz einfach, daß der Besitzer dieser Hand gar nicht tot war und daß er einfach gekommen ist, um sie mit der Hand wieder zu holen, die ihm übrig geblieben war ; aber ich weiß nicht, wie er das angestellt hat. Das wird eine Art Vendetta sein. Eine der Damen flüsterte : – Nein, das kann nicht so gewesen sein! Und der Untersuchungsrichter schloß immer noch lächelnd : – Ich habe es Ihnen doch gesagt, daß meine Erklärung Ihnen nicht passen würde."""
import unicodedata
def normalize_uroman(text):
    text = text.encode('utf-8').decode('utf-8')
    text = text.lower()
    text = text.replace("’", "'")
    text = unicodedata.normalize('NFC', text)
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()


#assert is_single_token(EVENT_MARKER)

# ─────────────────────────────────────────────────────────────
# 5)  MAIN LOOP
# ─────────────────────────────────────────────────────────────
idx_before = lambda W: [w-1 if w else 0 for w in W]
remove_punctuation = False



for lookahead in LOOKAHEAD:
    for name, text in stories.items():

        # 1) Normalize whitespace
        base_text = normalize_uroman(text) if remove_punctuation else text.strip()

        # 2) Words + original EB markers
        words, EB = strip_eb_markers(base_text)

        # 2b) Align rater EBs to canonical words
        HR_for_this_story = HR_dict
        EB_by_rater, raw_by_rater = {}, {}
        for rater_name, rater_text in HR_for_this_story.items():
            r_text = normalize_uroman(rater_text) if remove_punctuation else rater_text.strip()
            r_words, r_EB_raw = strip_eb_markers(r_text)
            EB_by_rater[rater_name] = align_indices(words, r_words, r_EB_raw)
            raw_by_rater[rater_name] = r_words

        # 3) DataSequence
        ds = make_ds(words, AVG_TR, WPM)

        # 4) Raw text for greedy (if used)
        raw_text = base_text.replace("¶", "")
        raw_text = re.sub(r"[ \t]+", " ", raw_text)
        raw_text = re.sub(r" *\n+ *", "\n\n", raw_text)

        # 5) Get boundary scores (GPU or cached)
        if run_model_on_gpu:
            results_tr, results_word, results_tok = [], [], []
            for _ in range(1):
                lp_tr_i, lp_word_i, lp_tok_i = boundary_lp(ds, tok, model, lookahead)
                results_tr.append(lp_tr_i); results_word.append(lp_word_i); results_tok.append(lp_tok_i)
            lp_tr   = sum(results_tr)   / len(results_tr)
            lp_word = sum(results_word) / len(results_word)
            lp_tok  = sum(results_tok)  / len(results_tok)
            np.savez('saved_LLM_results/results_tr.npz', results_tr)
            np.savez('saved_LLM_results/results_word.npz', results_word)
            np.savez('saved_LLM_results/results_tok.npz', results_tok)
        else:
            lp_tr   = np.load('saved_LLM_results/results_tr.npz')['arr_0'][0]
            lp_word = np.load('saved_LLM_results/results_word.npz')['arr_0'][0]
            lp_tok  = np.load('saved_LLM_results/results_tok.npz')['arr_0'][0]

        # 6) word→token map (for checks)
        tok_for_word = defaultdict(list)
        enc = tok(words, is_split_into_words=True, add_special_tokens=False)
        for t_idx, w_id in enumerate(enc.word_ids()):
            if w_id is not None:
                tok_for_word[w_id].append(t_idx)

        # rater token indices
        eb_tok_by_rater = {}
        for rater_name, EB_list in EB_by_rater.items():
            eb_tok_ = [last_or_none(tok_for_word[w - 1]) for w in EB_list]
            eb_tok_by_rater[rater_name] = [t for t in eb_tok_ if t is not None]

        # 7) Map EBs to TR & times
        eb_tr_by_rater = {
            rater_name: [np.argmin(np.abs(ds.tr_times - ds.data_times[w - 1])) for w in EB_list]
            for rater_name, EB_list in EB_by_rater.items()
        }

        EB_before = idx_before(EB)
        EB_before_by_rater = {rater_name: idx_before(EB_list) for rater_name, EB_list in EB_by_rater.items()}

        # 8) Security checks
        for rater_name, EB_list in EB_by_rater.items():
            for w in EB_list:
                assert 0 < w <= len(words), f"[{rater_name}] EB index {w} out of range 1..{len(words)}"
                assert tok_for_word[w - 1], f"[{rater_name}] tokens missing at word {w - 1}"
                if w < len(words):
                    assert tok_for_word[w], f"[{rater_name}] tokens missing at word {w}"

        recon = []
        for i in range(len(words) + 1):
            if i in EB: recon.append("¶")
            if i < len(words): recon.append(words[i])
        count_recon = " ".join(recon).count("¶")
        print(f"Reconstructed {count_recon} markers but expected {len(EB)})")
        print("✅ Security checks passed for", name)

        # ======== PAPER METRICS / PLOTS ========
        set_paper_style()
        TOL_WORDS = 0
        BLOCK_LEN = 50
        N_BOOT    = 10000
        TITLE_TAG = f"{name}, lookahead={lookahead}"
        OUT_PREFIX = f"{name}_la{lookahead}"

        # Labels (union across raters)
        y = build_labels_union(EB_by_rater, n_words=len(lp_word), tol_words=TOL_WORDS)

        punct = punctuation_flags(words).astype(int)
        mask = np.isfinite(lp_word)
        labels = y[mask].astype(int)
        score_llm = lp_word[mask]
        # --- CHECK 1: stratified AUROCs (beyond punctuation presence) ---
        m_non = (punct[mask] == 0)
        m_pun = (punct[mask] == 1)

        auc_llm_nonpunct = auroc(score_llm[m_non], labels[m_non]) if m_non.any() else np.nan
        auc_llm_punct = auroc(score_llm[m_pun], labels[m_pun]) if m_pun.any() else np.nan

        print(f"[LLM→EB | non-punct] AUROC={auc_llm_nonpunct:.3f}")
        print(f"[LLM→EB | punct-only] AUROC={auc_llm_punct:.3f}")
        # --- CHECK 2: residualize the punctuation mean shift and re-test ---
        s_resid = score_llm.copy()
        if m_non.any():
            s_resid[m_non] -= s_resid[m_non].mean()
        if m_pun.any():
            s_resid[m_pun] -= s_resid[m_pun].mean()

        auc_llm_resid = auroc(s_resid, labels)
        print(f"[LLM residual beyond punctuation mean shift → EB] AUROC={auc_llm_resid:.3f}")

        pos_idx = np.where(y==1)[0]; neg_idx = np.where(y==0)[0]

        # ROC/PR + metrics
        auc, auprc = plot_roc_pr(lp_word, y, TITLE_TAG, OUT_PREFIX)
        dprime_auc   = dprime_from_auc(auc)
        dprime_param = pooled_dprime(lp_word[pos_idx], lp_word[neg_idx])
        auc_boot, (auc_lo, auc_hi) = block_bootstrap_auc(lp_word, y, block_len=BLOCK_LEN, n_boot=N_BOOT, seed=0)

        # Distributions
        plot_score_distributions(lp_word, y, TITLE_TAG, f"{OUT_PREFIX}_score_dists")

        # Consensus curve with CIs + counts
        cons_info = plot_consensus_curve_with_ci(
            lp_word, EB_by_rater, TOL_WORDS, TITLE_TAG, OUT_PREFIX,
            block_len=BLOCK_LEN, n_boot=2000, seed=0
        )
        rho_consensus = spearman_corr(lp_word, consensus_counts(EB_by_rater, len(lp_word)))

        # Optional peaks operating point
        DIST_WORDS = 5
        PROM   = np.percentile(lp_word, 90) - np.percentile(lp_word, 50)
        HEIGHT = np.percentile(lp_word, 80)
        peaks  = find_peaks_nms(lp_word, min_distance=DIST_WORDS, min_height=HEIGHT, min_prom=PROM)

        true_union = sorted(set(np.where(build_labels_union(EB_by_rater, len(lp_word), tol_words=0)==1)[0]))
        hits, misses, fas = match_with_tolerance(peaks, true_union, tol=TOL_WORDS)
        crs = int(len(neg_idx) - fas)
        dprime_peaks, H_rate, FA_rate = dprime_from_counts(hits, misses, fas, crs)

        # Time-series panel with marks
        plot_timeseries_with_marks(ds, lp_word, EB_by_rater, peaks, TITLE_TAG, OUT_PREFIX)

        # Save metrics JSON + LaTeX captions
        metrics = {
            "story": name, "lookahead": int(lookahead),
            "n_words": int(len(lp_word)),
            "tolerance_words": int(TOL_WORDS),
            "block_bootstrap": {"block_len": BLOCK_LEN, "n_boot": N_BOOT},
            "AUC": float(auc), "AUC_boot_mean": float(auc_boot),
            "AUC_CI_95": [float(auc_lo), float(auc_hi)],
            "AUPRC": float(auprc),
            "dprime_from_AUC": float(dprime_auc),
            "dprime_parametric": float(dprime_param),
            "consensus_spearman_rho": float(rho_consensus),
            "peaks": {
                "distance_words": int(DIST_WORDS),
                "prominence": float(PROM),
                "height": float(HEIGHT),
                "hits": int(hits), "misses": int(misses),
                "false_alarms": int(fas), "correct_rejections": int(crs),
                "H_rate": float(H_rate), "FA_rate": float(FA_rate),
                "dprime": float(dprime_peaks)
            }
        }
        with open(f"out/{OUT_PREFIX}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


        print(
            f"[{TITLE_TAG}] "
            f"AUC={auc:.3f} (boot {auc_boot:.3f}, 95% CI [{auc_lo:.3f},{auc_hi:.3f}]), "
            f"AUPRC={auprc:.3f}, d′(AUC)={dprime_auc:.3f}, d′(param)={dprime_param:.3f}; "
            f"peaks d′={dprime_peaks:.3f} (H={H_rate:.3f}, FA={FA_rate:.3f}); "
            f"ρ_consensus={rho_consensus:.3f}"
        )

        # 9) Token map on ds.data for token-level plots
        ds_tok_for_word = {w: [] for w in range(len(ds.data))}
        enc_ds = tok(ds.data.tolist(), is_split_into_words=True, add_special_tokens=False)
        for i_tok, w_id in enumerate(enc_ds.word_ids()):
            if w_id is not None:
                ds_tok_for_word[w_id].append(i_tok)
        eb_tok_by_rater_ds = {
            rater_name: [max(ds_tok_for_word[w - 1]) for w in EB_list if ds_tok_for_word.get(w - 1)]
            for rater_name, EB_list in EB_by_rater.items()
        }

        # ===== Paper-ready TR / WORD / TOKEN summary figures (drop-in) =====
        # ===== Paper-ready TR / WORD / TOKEN summary figures (12x3) =====
        set_paper_style()  # keep global style; override figsize below

        TITLE_SHORT = f"{name}, la={lookahead}"  # not shown (no title requested)

        markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*', 'h', '+']
        colors = [COL["orange"], COL["green"], COL["red"], COL["purple"],
                  COL["sky"], COL["yellow"], COL["black"], COL["grey"],
                  "#8dd3c7", "#bebada", "#fb8072", "#80b1d3"]

        # --- TR-level (12x3), y-label as log(prob), raters shown as R1, R2, ...
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(ds.tr_times, lp_tr, lw=1.4, color=COL["blue"], label="log(prob)")
        for i, (rater_name, eb_tr) in enumerate(eb_tr_by_rater.items(), start=1):
            if len(eb_tr):
                ax.scatter(ds.tr_times[eb_tr], lp_tr[eb_tr],
                           s=12, marker=markers[(i - 1) % len(markers)],
                           color=colors[(i - 1) % len(colors)], alpha=0.95,
                           label=f"R{i}")
        ax.set(xlabel="time (s)", ylabel="log(prob)")  # no title
        _prettify_axes(ax)
        ax.legend(frameon=False, ncol=6, loc="best")
        _savefig_pair(f"out/feature_summary_{name}_TR_fixedlokkaheadandmarkerpos_lookahead_{lookahead}4R_MEAN_NOCACHE")

        # --- WORD-level (12x3), remove 'orig EB' & smoothing; raters as R1, R2, ...
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(ds.data_times, lp_word, lw=0.74, color=COL["blue"], alpha=0.95)
        for i, (rater_name, EB_before_r) in enumerate(EB_before_by_rater.items(), start=1):
            if len(EB_before_r):
                ax.scatter(ds.data_times[EB_before_r], lp_word[EB_before_r],
                           s=14, marker=markers[(i - 1) % len(markers)],
                           color=colors[(i - 1) % len(colors)], alpha=0.95,
                           label=f"R{i}")
        ax.set(xlabel="time (s)", ylabel="log(prob)")  # no title
        _prettify_axes(ax)
        ax.legend(frameon=False, ncol=6, loc="best")
        _savefig_pair(f"out/feature_summary_{name}_WORD_fixedlokkaheadandmarkerpos_lookahead_{lookahead}4R_MEAN_NOCACHE")

        # --- TOKEN-level (12x3), raters as R1, R2, ...
        fig, ax = plt.subplots(figsize=(12, 3))
        x_idx = np.arange(len(lp_tok))
        ax.plot(x_idx, lp_tok, lw=1.0, color=COL["blue"], label="log(prob)")
        for i, (rater_name, eb_tok) in enumerate(eb_tok_by_rater_ds.items(), start=1):
            if len(eb_tok):
                ax.scatter(eb_tok, [lp_tok[j] for j in eb_tok],
                           s=10, marker=markers[(i - 1) % len(markers)],
                           color=colors[(i - 1) % len(colors)], alpha=0.95,
                           label=f"R{i}")
        ax.set(xlabel="token index", ylabel="log(prob)")  # no title
        _prettify_axes(ax)
        ax.legend(frameon=False, ncol=6, loc="best")
        _savefig_pair(f"out/feature_summary_{name}_TOK_fixedlokkaheadandmarkerpos_lookahead_{lookahead}4R_MEAN_NOCACHE")


        #### IS IT JUST PUNCTUATION ????#####
        # ===== Punctuation vs LLM boundary score =====
        set_paper_style()

        # Build punctuation indicator aligned to word boundaries
        punct = punctuation_flags(words)  # shape (n_words,)
        mask = np.isfinite(lp_word)  # safety mask in case of NaNs
        binar = punct[mask].astype(int)
        scores = lp_word[mask]

        # Stats: point-biserial correlation & AUROC (punctuation as "positive")
        r_pb = float(np.corrcoef(binar, scores)[0, 1])
        auc_punct = auroc(scores, binar)  # uses your auroc helper

        n_pos = int(binar.sum())
        n_all = int(binar.size)
        m_pos = float(scores[binar == 1].mean()) if n_pos > 0 else float('nan')
        m_neg = float(scores[binar == 0].mean()) if n_pos < n_all else float('nan')

        print(f"[punctuation] n={n_pos}/{n_all} ({n_pos / n_all:.3f})  "
              f"r_pb={r_pb:.3f}  AUROC={auc_punct:.3f}  "
              f"mean_punct={m_pos:.3f}  mean_non={m_neg:.3f}")

        # Figure: distributions of log(prob) for punctuation vs no-punctuation
        fig, ax = plt.subplots(figsize=(12, 3))

        # tight, symmetric range for readability
        lo, hi = np.quantile(scores, [0.002, 0.998])
        bins = max(20, int(np.sqrt(len(scores))))

        ax.hist(scores[binar == 0], bins=bins, range=(lo, hi), density=True,
                color=COL.get("blue", "#1f77b4"), alpha=0.35, edgecolor=COL.get("blue", "#1f77b4"),
                linewidth=0.7, label="no punctuation")
        ax.hist(scores[binar == 1], bins=bins, range=(lo, hi), density=True,
                color=COL.get("orange", "#ff7f0e"), alpha=0.35, edgecolor=COL.get("orange", "#ff7f0e"),
                linewidth=0.7, label="punctuation")

        # Add medians
        if n_pos:
            ax.axvline(np.median(scores[binar == 1]), color=COL.get("orange", "#ff7f0e"), lw=1.2)
        if n_pos < n_all:
            ax.axvline(np.median(scores[binar == 0]), color=COL.get("blue", "#1f77b4"), lw=1.2)

        # Annotate with quick stats
        txt = (f"r (point-biserial) = {r_pb:.3f}\n"
               f"AUROC(punct) = {auc_punct:.3f}\n"
               f"n_punct = {n_pos} / {n_all}")
        try:
            _prettify_axes(ax)
            # if you added the friendly _annot earlier:
            try:
                _annot(ax, txt)
            except:
                ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, color="#222",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="#BBBBBB", linewidth=0.6, alpha=0.9))
        except:
            pass

        ax.set(xlabel="log(prob)", ylabel="density")  # your y-axis is log-prob already
        ax.legend(frameon=False, ncol=2, loc="best")

        # Save both PNG and PDF (uses your helper if present)
        try:
            _savefig_pair(f"out/{name}_punct_vs_boundary")
        except:
            #plt.tight_layout()
            plt.savefig(f"out/{name}_punct_vs_boundary.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"out/{name}_punct_vs_boundary.pdf", bbox_inches="tight")
            plt.close()

        # ===== Punctuation as a baseline predictor of human EBs =====
        set_paper_style()

        # 1) Binary punctuation flag per word
        punct = punctuation_flags(words).astype(int)  # 1 if word ends with . ! ? … ; : , (after closers), else 0
        mask = np.isfinite(lp_word)  # safety (should be all True)
        scores_llm = lp_word[mask]
        labels_eb = y[mask]  # human EB union (0/1)
        scores_punct = punct[mask]  # use punctuation 0/1 as "score"

        # 2) AUROC & d′: punctuation-only vs human EBs
        auc_punct_vs_eb = auroc(scores_punct, labels_eb)  # works with 0/1 scores
        dprime_punct_auc = dprime_from_auc(auc_punct_vs_eb)

        # 3) PR for punctuation-only (baseline is prevalence)
        rec_p, prec_p, auprc_punct = pr_curve_from_scores(scores_punct, labels_eb)
        prev = float(labels_eb.mean())  # PR baseline

        # 4) For comparison: AUROC of your LLM score vs human EBs (you already computed earlier)
        # If you have the variable 'auc' from plot_roc_pr(lp_word, y, ...), reuse it; otherwise recompute:
        auc_llm_vs_eb = auroc(scores_llm, labels_eb)
        dprime_llm_auc = dprime_from_auc(auc_llm_vs_eb)

        print(
            f"[punct→EB] AUROC={auc_punct_vs_eb:.3f}  d′(AUC)={dprime_punct_auc:.3f}  "
            f"AUPRC={auprc_punct:.3f} (baseline={prev:.3f})  "
            f"|  [LLM→EB] AUROC={auc_llm_vs_eb:.3f}  d′(AUC)={dprime_llm_auc:.3f}"
        )

        # 5) Small bar plot: AUROC (LLM vs punctuation) — paper-ready 12x3
        fig, ax = plt.subplots(figsize=(12, 3))
        x = np.arange(2)
        vals = [auc_llm_vs_eb, auc_punct_vs_eb]
        ax.bar(x, vals, color=[COL.get("blue", "#1f77b4"), COL.get("orange", "#ff7f0e")], alpha=0.9, width=0.6)
        ax.set_xticks(x, ["LLM score → EB", "Punctuation → EB"])
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUROC")
        _prettify_axes(ax)
        # optional annotation
        try:
            _annot(ax, f"d′(LLM)={dprime_llm_auc:.2f}\n"
                       f"d′(Punct)={dprime_punct_auc:.2f}")
        except:
            ax.text(x=0.98, y=0.98, s=f"d′(LLM)={dprime_llm_auc:.2f}\n"
                                      f"d′(Punct)={dprime_punct_auc:.2f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
        try:
            _savefig_pair(f"out/{name}_auc_LLMeB_vs_PuncteB")
        except:
            plt.tight_layout()
            plt.savefig(f"out/{name}_auc_LLMeB_vs_PuncteB.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"out/{name}_auc_LLMeB_vs_PuncteB.pdf", bbox_inches="tight")
            plt.close()

        # 6) Optional: PR curve for punctuation-only (to see how it sits vs baseline)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(rec_p, prec_p, color=COL.get("orange", "#ff7f0e"), label="punctuation-only PR")
        ax.axhline(prev, ls="--", lw=1.0, color=COL.get("grey", "#888888"), label=f"baseline={prev:.3f}")
        ax.set(xlabel="recall", ylabel="precision")
        _prettify_axes(ax)
        ax.legend(frameon=False, loc="best")
        try:
            _savefig_pair(f"out/{name}_PR_punctuation_only")
        except:
            plt.tight_layout()
            plt.savefig(f"out/{name}_PR_punctuation_only.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"out/{name}_PR_punctuation_only.pdf", bbox_inches="tight")
            plt.close()

