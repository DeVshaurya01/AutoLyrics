"""One-shot script to overwrite data/interim/lyrics/*.txt with full song lyrics.

Usage:  python scripts/_fill_lyrics.py
Safe to delete after running.
"""
import re
from pathlib import Path

LYRICS = {
"01": """Edelweiss, edelweiss
Every morning you greet me
Small and white, clean and bright
You look happy to meet me
Blossom of snow may you bloom and grow
Bloom and grow forever
Edelweiss, edelweiss
Bless my homeland forever
Edelweiss, edelweiss
Every morning you greet me
Small and white, clean and bright
You look happy to meet me
Blossom of snow may you bloom and grow
Bloom and grow forever
Edelweiss, edelweiss
Bless my homeland forever""",

"02": """Let's start at the very beginning
A very good place to start
When you read, you begin with
A-B-C
When you sing, you begin with do-re-mi
Do-re-mi
Do-re-mi
The first three notes just happen to be
Do-re-mi
Do-re-mi
Do-re-mi-fa-so-la-ti
Oh, let's see if I can make it easier
You might also like
Port Antonio
J. Cole
THE HEART PART 6
Drake
SKINNY
Billie Eilish
Do, a deer, a female deer
Re, a drop of golden sun
Mi, a name I call myself
Fa, a long, long way to run
So, a needle pulling thread
La, a note to follow so
Ti, a drink with jam and bread
That will bring us back to do, oh, oh, oh
Do
A deer, a female deer
Re
A drop of golden sun
Mi
A name I call myself
Fa
A long, long way to run
So, a needle pulling thread
La
A note to follow so
Ti
A drink with jam and bread
That will bring us back to do
Do-re-mi-fa-so-la-ti-do, so-do
Now children, do, re, mi, fa, so, and so on are only the tools we use to build a song
Once you have these notes in your heads, you can sing a million different tunes by mixing them up like this
So, do, la, fa, mi, do, re
Can you do that
So, do, la, fa, mi, do, re
So, do, la, ti, do, re, do
So, do, la, ti, do, re, do
Now put it all together
So, do, la, fa, mi, do, re
So, do, la, ti, do, re, do
Good
But it doesn't mean anything
So we put in words, one word for every note, like this
When you know the notes to sing
You can sing most anything
Together
When you know the notes to sing
You can sing most anything
Do, a deer, a female deer
Re, a drop of golden sun
Mi, a name I call myself
Fa, a long, long way to run
So, a needle pulling thread
La, a note to follow so
Ti, a drink with jam and bread
That will bring us back to do""",

"03": """Dashing through the snow
In a one-horse open sleigh
Over the hills we go
Laughing all the way
The bells on bobtail ring
They make spirits bright
What fun it is to ride and sing a sleighing song tonight
Jingle bells, jingle bells
Jingle all the way
What fun it is to ride
In a one horse open sleigh, hey
Jingle bells, jingle bells
Jingle all the way
Oh, what fun it is to ride
In a one horse open sleigh
A day or two ago
I thought I'd take a ride
And soon Miss Fanny Bright
Was seated by my side
The horse was lean and lank
Misfortune seemed his lot
We got into a drifted bank
And we will got upsot
You might also like
Are You Gone Already
Nicki Minaj
2024
Playboi Carti
ino quintero
Genius Deutsche Uebersetzungen
Jingle bells, jingle bells
Jingle all the way
Oh, what fun it is to ride
In a one horse open sleigh, hey
Jingle bells, jingle bells
Jingle all the way
Oh, what fun it is to ride
In a one horse open sleigh
A day or two ago
The story I must tell
I went out on the snow
And on my back I fell
A gent was riding by
In a one-horse open sleigh
He laughed as there I sprawling lie
But quickly drove away
Jingle bells, jingle bells
Jingle all the way
Oh, what fun it is to ride
In a one horse open sleigh, hey
Jingle bells, jingle bells
Jingle all the way
Oh, what fun it is to ride
In a one horse open sleigh
Now the ground is white
To go while we are young
Take the girls tonight
And sing this sleighing song
Just get a bobtailed bay
Two-forty is his speed
Then hitch him to an open sleigh
And crack you will take the lead, oh
Jingle bells, jingle bells
Jingle all the way
Oh, what fun it is to ride
In a one horse open sleigh, hey
Jingle bells, jingle bells
Jingle all the way
Oh, what fun it is to ride
In a one horse open sleigh""",

"04": """Silent night, holy night
All is calm, all is bright
Round yon Virgin, Mother and Child
Holy Infant so tender and mild
Sleep in heavenly peace
Sleep in heavenly peace
Silent night, holy night
Shepherds quake at the sight
Glories stream from heaven afar
Heavenly hosts sing Alleluia
Christ the Savior is born
Christ the Savior is born
Silent night, holy night
Son of God, love's pure light
Radiant beams from Thy holy face
With the dawn of redeeming grace
Jesus Lord, at Thy birth
Jesus Lord, at Thy birth""",

"05": """It's late in the evening
She's wondering what clothes to wear
She puts on her make up
And brushes her long blonde hair
And then she asks me
Do I look alright
And I say yes you look wonderful tonight
We go to a party
And everyone turns to see
This beautiful lady
That's walking around with me
And then she asks me
Do you feel alright
And I say yes I feel wonderful tonight
I feel wonderful
Because I see the love light in your eyes
And the wonder of it all
Is that you just don't realize how much I love you
It's time to go home now
And I've got an aching head
So I give her the car keys
And she helps me to bed
And then I tell her
As I turn out the light
I said my darling you are wonderful tonight
Oh my darling you are wonderful tonight""",

"06": """Moon river, wider than a mile
I'm crossing you in style some day
Oh, dream maker, you heartbreaker
Wherever you're goin', I'm goin' your way
Two drifters, off to see the world
There's such a lot of world to see
We're after the same rainbow's end
Waitin' 'round the bend, my huckleberry friend
Moon river and me""",

"07": """Listen to the rhythm of the falling rain
Telling me just what a fool I've been
I wish that it would go and let me cry in vain
And let me be alone again
Now the only girl I've ever loved has gone away
Looking for a brand new start
But little does she know that when she left that day
Along with her she took my heart
Rain, please tell me, now does that seem fair
For her to steal my heart away when she don't care
I can't love another, when my heart's somewhere far away
Rain, won't you tell her that I love her so
Please ask the sun to set her heart aglow
Rain in her heart and let the love we knew start to grow""",

"08": """I have a dream, a song to sing
To help me cope with anything
If you see the wonder of a fairy tale
You can take the future even if you fail
I believe in angels
Something good in everything I see
I believe in angels
When I know the time is right for me
I'll cross the stream, I have a dream
I have a dream, a fantasy
To help me through reality
And my destination, makes it worth the while
Pushin' through the darkness, still another mile
I believe in angels
Something good in everything I see
I believe in angels
When I know the time is right for me
I'll cross the stream, I have a dream
I'll cross the stream, I have a dream
I have a dream, a song to sing
To help me cope with anything
If you see the wonder of a fairy tale
You can take the future even if you fail
I believe in angels
Something good in everything I see
I believe in angels
When I know the time is right for me
I'll cross the stream, I have a dream
I'll cross the stream, I have a dream""",

"09": """Love me tender, love me sweet
Never let me go
You have made my life complete
And I love you so
Love me tender, love me true
All my dreams fulfill
For, my darling, I love you
And I always will
Love me tender, love me long
Take me to your heart
For it's there that I belong
And we'll never part
Love me tender, love me true
All my dreams fulfill
For, my darling, I love you
And I always will
Love me tender, love me dear
Tell me you are mine
I'll be yours through all the years
Till the end of time
You might also like
But Daddy I Love Him
Taylor Swift
The Tortured Poets Department
Taylor Swift
So Long, London
Taylor Swift
Love me tender, love me true
All my dreams fulfill
For, my darling, I love you
And I always will""",

"10": """Twinkle, twinkle, little star
How I wonder what you are
Up above the world so high
Like a diamond in the sky
Twinkle, twinkle, little star
How I wonder what you are
When the blazing sun is gone
When he nothing shines upon
Then you show your little light
Twinkle, twinkle, all the night
Twinkle, twinkle, little star
How I wonder what you are
Then the traveller in the dark
Thanks you for your tiny spark
He could not see which way to go
If you did not twinkle so
Twinkle, twinkle, little star
How I wonder what you are
In the dark blue sky you keep
And often through my curtains peep
For you never shut your eye
Till the sun is in the sky
Twinkle, twinkle, little star
How I wonder what you are
As your bright and tiny spark
Lights the traveller in the dark
Though I know not what you are
Twinkle, twinkle, little star
Twinkle, twinkle, little star
How I wonder what you are""",

"11": """The other night dear as I lay sleeping
I dreamt I held you in my arms
But when I woke dear I was mistaken
And I hung my head and cried
You are my sunshine my only sunshine
You make me happy when skies are gray
You'll never know dear how much I love you
Please don't take my sunshine away
I'll always love you and make you happy
If you will only say the same
But if you leave me to love another
You'll regret it all someday
You are my sunshine my only sunshine
You make me happy when skies are gray
You'll never know dear how much I love you
Please don't take my sunshine away
You told me once dear you really loved me
That no one else could come between
But now you've left me and love another
You have shattered all my dreams
You are my sunshine my only sunshine
You make me happy when skies are gray
You'll never know dear how much I love you
Please don't take my sunshine away""",

"12": """Greatness as you
Smallest as me
You show me what is deep as sea
A little love, little kiss
A little hug, little gift
All of little something, these are our memories
You make me cry
Make me smile
Make me feel that love is true
You always stand by my side
I don't want to say goodbye
You make me cry
Make me smile
Make me feel the joy of love
Oh kissing you
Thank you for all the love you always give to me
Oh I love you
Greatness as you
Smallest as me
You show me what is deep as sea
A little love, little kiss
A little hug, little gift
All of little something, these are our memories
You make me cry
Make me smile
Make me feel that love is true
You always stand by my side
I don't want to say goodbye
You make me cry
Make me smile
Make me feel the joy of love
Oh kissing you
Thank you for all the love you always give to me
Oh I love you
Yes I do, I always do
I always do
Make me cry
Make me smile
Make me feel that love is true
You always stand by my side
I don't want to say goodbye
You make me cry
Make me smile
Make me feel the joy of love
Oh kissing you
Thank you for all the love you always give to me
Oh I love you
To be with you
Oh I love you""",

"13": """Love in your eyes, sitting silent by my side
Going on, holding hands, walking through the nights
Hold me up, hold me tight, lift me up to touch the sky
Teaching me to love with heart, helping me open my mind
I can fly, I'm proud that I can fly
To give the best of mine, till the end of the time
Believe me I can fly, I'm proud that I can fly
To give the best of mine, the heaven's in the sky
Stars in the sky, wishing once upon the time
Give me love, make me smile, till the end of life
Hold me up, hold me tights, lift me up to touch the sky
Teaching me to love with heart, helping me open my mind
I can fly, I'm proud that I can fly
To give the best of mine, till the end of the time
Believe me I can fly, I'm proud that I can fly
To give the best of mine, the heaven's in the sky
Can't you believe that you light up my way
No matter how that ease my path
I'll never lose my faith
You might also like
Salamat
Yeng Constantino
SEVENTEEN
Genius English Translations
Despacito
Luis Fonsi
See me fly, I'm proud to fly up high
Show you the best of mine, till the end of the time
Believe me I can fly
I'm singing in the sky
Show you the best of mine
The heaven in the sky
Nothing can stop me spread my wings so wide""",

"14": """I'm sitting here in the boring room
It's just another rainy Sunday afternoon
I'm wasting my time
I got nothing to do
I'm hanging around
I'm waiting for you
But nothing ever happens and I wonder
I'm driving around in my car
I'm driving too fast
I'm driving too far
I'd like to change my point of view
I feel so lonely
I'm waiting for you
But nothing ever happens and I wonder
I wonder how
I wonder why
Yesterday you told me 'bout the blue, blue sky
And all that I can see is just a yellow lemon tree
I'm turning my head up and down
I'm turning, turning, turning, turning, turning around
And all that I can see is just another lemon tree
Sing
I'm sitting here
I miss the power
I'd like to go out taking a shower
But there's a heavy cloud inside my head
I feel so tired
Put myself into bed
While nothing ever happens and I wonder
Isolation is not good for me
Isolation I don't want to sit on the lemon tree
I'm stepping around in the desert of joy
Baby anyhow I'll get another toy
And everything will happen and you wonder
I wonder how
I wonder why
Yesterday you told me 'bout the blue, blue sky
And all that I can see is just another lemon tree
I'm turning my head up and down
I'm turning, turning, turning, turning, turning around
And all that I can see is just a yellow lemon tree
And I wonder, wonder
I wonder how
I wonder why
Yesterday you told me 'bout the blue, blue sky
And all that I can see, and all that I can see, and all that I can see
Is just a yellow lemon tree""",

"15": """There's a calm surrender to the rush of day
When the heat of a rolling wind can be turned away
An enchanted moment, and it sees me through
It's enough for this restless warrior just to be with you
Can you feel the love tonight
It is where we are
It's enough for this wide-eyed wanderer
That we've got this far
And can you feel the love tonight
How it's laid to rest
Oh, it's enough to make kings and vagabonds
Believe the very best
There's a time for everyone if they only learn
That the twisting kaleidoscope moves us all in turn
There's a rhyme and reason to the wild outdoors
When the heart of this star-crossed voyager beats in time with yours
Can you feel the love tonight
It is where we are
It's enough for this wide-eyed wanderer
That we've got this far
Can you feel the love tonight
How it's laid to rest
Oh, it's enough to make kings and vagabonds
Believe the very best
Oh, it's enough to make kings and vagabonds
Believe the very best""",

"16": """I'm loving living every single day but sometimes I feel so
I hope to find a little piece of mind and I just want to know
And who can heal those tiny broken hearts, and what are we to be
Where is home on the milkyway of stars, I dry my eyes again
In my dreams I am not so far away from home
What am I in a world so far away from home
All my life all the time so far away from home
Without you I will be so far away from home
If we could make it through the darkest night we'd have a brighter day
The world I see beyond your pretty eyes, makes me want to stay
And who can heal those tiny broken hearts, and what are we to be
Where is home on the milkyway of stars, I dry my eyes again
In my dreams I am not so far away from home
What am I in a world so far away from home
All my life all the time so far away from home
Without you I will be so far away from home
I count on you, no matter what they say, cause love can find its time
I hope to be a part of you again, baby let us shine
And who can heal those tiny broken hearts, and what are we to be
Where is home on the milkyway of stars, I dry my eyes again
In my dreams I am not so far away from home
What am I in a world so far away from home
All my life all the time so far away from home
Without you I will be so far away from home
In my dreams I am not so far away from home
What am I in a world so far away from home
All my life all the time so far away from home
Without you I will be so far away from home""",

"17": """Goodbye to you, my trusted friend
We've known each other since we were nine or ten
Together we've climbed hills and trees
Learned of love and ABCs
Skinned our hearts and skinned our knees
Goodbye my friend, it's hard to die
When all the birds are singing in the sky
Now that the spring is in the air
Pretty girls are everywhere
Think of me and I'll be there
We had joy, we had fun
We had seasons in the sun
But the hills that we climbed
Were just seasons out of time
Goodbye papa, please pray for me
I was the black sheep of the family
You tried to teach me right from wrong
Too much wine and too much song
Wonder how I got along
Goodbye papa, it's hard to die
When all the birds are singing in the sky
Now that the spring is in the air
Little children everywhere
When you see them, I'll be there
We had joy, we had fun
We had seasons in the sun
But the wine and the song
Like the seasons, have all gone
We had joy, we had fun
We had seasons in the sun
But the wine and the song
Like the seasons, have all gone
Goodbye Michelle, my little one
You gave me love and helped me find the sun
And every time that I was down
You would always come around
And get my feet back on the ground
Goodbye Michelle, it's hard to die
When all the birds are singing in the sky
Now that the spring is in the air
With the flowers everywhere
I wish that we could both be there
We had joy, we had fun
We had seasons in the sun
But the stars we could reach
Were just starfish on the beach
We had joy, we had fun
We had seasons in the sun
But the stars we could reach
Were just starfish on the beach
We had joy, we had fun
We had seasons in the sun
But the wine and the song
Like the seasons, have all gone
All our lives we had fun
We had seasons in the sun
But the hills that we climbed
Were just seasons out of time""",

"18": """I'm just a little bit caught in the middle
Life is a maze, and love is a riddle
I don't know where to go, can't do it alone
I've tried, and I don't know why
Slow it down, make it stop
Or else my heart is going to pop
'Cause it's too much, yeah, it's a lot
To be something I'm not
I'm a fool, out of love
'Cause I just can't get enough
I'm just a little bit caught in the middle
Life is a maze, and love is a riddle
I don't know where to go, can't do it alone
I've tried, and I don't know why
I'm just a little girl lost in the moment
I'm so scared, but I don't show it
I can't figure it out, it's bringing me down, I know
I've got to let it go
And just enjoy the show
The sun is hot in the sky
Just like a giant spotlight
The people follow the signs
And synchronize in time
It's a joke, nobody knows
They got a ticket to the show
Yeah, I'm just a little bit caught in the middle
Life is a maze, and love is a riddle
I don't know where to go, can't do it alone
I've tried, and I don't know why
I'm just a little girl lost in the moment
I'm so scared, but I don't show it
I can't figure it out, it's bringing me down, I know
I've got to let it go
And just enjoy the show
Just enjoy the show
I'm just a little bit caught in the middle
Life is a maze, and love is a riddle
I don't know where to go, can't do it alone
I've tried, and I don't know why
I'm just a little girl lost in the moment
I'm so scared, but I don't show it
I can't figure it out, it's bringing me down, I know
I've got to let it go
And just enjoy the show
Just enjoy the show
Just enjoy the show
I want my money back
I want my money back
I want my money back
Just enjoy the show
I want my money back
I want my money back
I want my money back
Just enjoy the show""",

"19": """Some say love, it is a river
That drowns the tender reed
Some say love, it is a razor
That leaves your soul to bleed
Some say love, it is a hunger
An endless aching need
I say love, it is a flower
And you, its only seed
It's the heart afraid of breaking
That never learns to dance
It's the dream afraid of waking
That never takes the chance
It's the one who won't be taking
Who cannot seem to give
And the soul, afraid of dying
That never learns to live
When the night has been too lonely
And the road has been too long
And you think that love is only
For the lucky and the strong
Just remember in the winter
Far beneath the bitter snows
Lies the seed that with the sun's love
In the spring becomes the rose""",

"20": """Oceans apart, day after day
And I slowly go insane
I hear your voice on the line
But it doesn't stop the pain
If I see you next to never
Then how can we say forever
Wherever you go
Whatever you do
I will be right here waiting for you
Whatever it takes
Or how my heart breaks
I will be right here waiting for you
I took for granted, all the times
That I thought would last somehow
I hear the laughter, I taste the tears
But I can't get near you now
Oh, can't you see it, baby
You've got me going crazy
Wherever you go
Whatever you do
I will be right here waiting for you
Whatever it takes
Or how my heart breaks
I will be right here waiting for you
I wonder how we can survive
This romance
But in the end if I'm with you
I'll take the chance
Oh, can't you see it, baby
You've got me going crazy
Wherever you go
Whatever you do
I will be right here waiting for you
Whatever it takes
Or how my heart breaks
I will be right here waiting for you
Waiting for you""",
}


def strip_genius_ads(text: str) -> str:
    """Drop 'You might also like' marker and the 6 lines after it (3 song/artist pairs)."""
    lines = text.split("\n")
    out = []
    skip = 0
    for line in lines:
        if skip > 0:
            skip -= 1
            continue
        if "you might also like" in line.lower():
            skip = 6
            continue
        out.append(line)
    return "\n".join(out)


def normalize(text: str) -> str:
    text = strip_genius_ads(text)
    text = text.lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)   # [Verse], [Chorus]
    text = re.sub(r"\([^)]*\)", " ", text)    # (spoken), (sung)
    text = text.replace("-", " ")             # do-re-mi -> do re mi
    text = re.sub(r"[^a-z'\s]", " ", text)    # strip non-letters/apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    out = Path("data/interim/lyrics")
    for sid in sorted(LYRICS):
        norm = normalize(LYRICS[sid])
        (out / f"{sid}.txt").write_text(norm + "\n", encoding="utf-8")
        print(f"{sid}.txt: {len(norm.split()):4d} words")
    print("\nAll 20 lyrics files written.")


if __name__ == "__main__":
    main()
